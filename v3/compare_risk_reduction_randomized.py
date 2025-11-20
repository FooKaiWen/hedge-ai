import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

# Assuming main_risk_reduction.py is in the same directory or accessible
from main_risk_reduction import HedgeEnvV3, load_full_dataset

# Number of times to run the evaluation to get a stable average
N_EVALUATIONS = 50

def calculate_metrics(pnl):
    """Calculates variance, Sharpe ratio, and cumulative PnL."""
    if pnl is None or len(pnl) == 0:
        return 0, 0, 0
    variance = np.var(pnl)
    sharpe_ratio = np.mean(pnl) / (np.std(pnl) + 1e-9)
    cumulative_pnl = np.sum(pnl)
    return variance, sharpe_ratio, cumulative_pnl

def evaluate_strategy(df_test, strategy, model=None, vec_normalize_path=None, features=None, seed=None):
    """Evaluates a given hedging strategy on the test set."""
    pnl_history = []
    hedge_ratios = []

    if strategy == 'rl':
        env_raw = DummyVecEnv([lambda: HedgeEnvV3(df_test, features=features)])
        # Important: Seed the environment for this specific run
        if seed is not None:
            env_raw.seed(seed)
        env = VecNormalize.load(vec_normalize_path, env_raw)
        env.training = False
        env.norm_reward = False
        
        obs = env.reset()
        for _ in range(len(df_test) - 91):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, info = env.step(action)
            pnl_history.append(info[0]['pnl'])
            hedge_ratios.append(env.envs[0].current_hedge_ratio)
    
    else: # Benchmark strategies
        current_hedge = 0
        if strategy == 'full':
            current_hedge = 1.0
        elif strategy == 'ols':
            current_hedge = 0.92 

        hedge_ratios = [current_hedge] * (len(df_test) - 1)
        spot_returns = df_test['spot_ret'].iloc[:-1].values
        futures_returns = df_test['fut_ret'].iloc[:-1].values
        pnl_history = spot_returns - current_hedge * futures_returns

    return pnl_history, hedge_ratios

def main():
    # --- 1. Load Data and Model ---
    df_full = load_full_dataset()
    train_size = int(len(df_full) * 0.8)
    df_test = df_full.iloc[train_size:].reset_index(drop=True)

    model_path = 'v3/models/ppo_hedge_risk_reduction_20251118_223523.zip'
    vec_norm_path = 'v3/models/vec_normalize_risk_reduction_20251118_223523.pkl'
    features_path = 'v3/models/features_risk_reduction_20251118_223523.json'

    if not all(os.path.exists(p) for p in [model_path, vec_norm_path, features_path]):
        print("Error: Model, normalization stats, or features file not found.")
        print("Please run 'main_risk_reduction.py' to train the model first.")
        return

    model = PPO.load(model_path)
    with open(features_path, 'r') as f:
        features = json.load(f)

    # --- 2. Evaluate All Strategies ---
    # Monte Carlo evaluation for the RL Agent
    all_pnl_rl = []
    all_ratios_rl = []
    print(f"Evaluating RL Agent over {N_EVALUATIONS} runs...")
    for i in tqdm(range(N_EVALUATIONS)):
        pnl_rl, ratios_rl = evaluate_strategy(df_test, 'rl', model, vec_norm_path, features, seed=i)
        all_pnl_rl.append(pnl_rl)
        all_ratios_rl.append(ratios_rl)
    
    # Calculate mean and std dev for RL agent's performance
    mean_pnl_rl = np.mean(all_pnl_rl, axis=0)
    std_pnl_rl = np.std(all_pnl_rl, axis=0)
    mean_ratios_rl = np.mean(all_ratios_rl, axis=0)
    std_ratios_rl = np.std(all_ratios_rl, axis=0)

    # Evaluate deterministic benchmarks once
    print("Evaluating benchmark strategies...")
    pnl_no_hedge, _ = evaluate_strategy(df_test, 'none')
    pnl_full_hedge, _ = evaluate_strategy(df_test, 'full')
    pnl_ols_hedge, ratios_ols = evaluate_strategy(df_test, 'ols')

    # --- 3. Calculate Average Metrics ---
    var_rl, sharpe_rl, cum_pnl_rl = calculate_metrics(mean_pnl_rl)
    var_no_hedge, sharpe_no_hedge, cum_pnl_no_hedge = calculate_metrics(pnl_no_hedge)
    var_full_hedge, sharpe_full_hedge, cum_pnl_full_hedge = calculate_metrics(pnl_full_hedge)
    var_ols_hedge, sharpe_ols_hedge, cum_pnl_ols_hedge = calculate_metrics(pnl_ols_hedge)

    vre_rl = 1 - (var_rl / var_no_hedge)
    vre_full_hedge = 1 - (var_full_hedge / var_no_hedge)
    vre_ols_hedge = 1 - (var_ols_hedge / var_no_hedge)

    # --- 4. Display Results Table ---
    results = {
        "Strategy": ["RL Agent (Avg)", "No Hedge (h=0.00)", "Full Hedge (h=1.00)", "OLS Hedge (h=0.92)"],
        "Variance": [var_rl, var_no_hedge, var_full_hedge, var_ols_hedge],
        "VRE": [vre_rl, 0.0, vre_full_hedge, vre_ols_hedge],
        "Sharpe Ratio": [sharpe_rl, sharpe_no_hedge, sharpe_full_hedge, sharpe_ols_hedge],
        "Cumulative PnL": [cum_pnl_rl, cum_pnl_no_hedge, cum_pnl_full_hedge, cum_pnl_ols_hedge]
    }
    results_df = pd.DataFrame(results)
    print("\n--- Average Risk & Return Performance on Test Set ---")
    print(results_df.to_string(index=False))

    # --- 5. Generate Comparison Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Cumulative PnL
    cum_pnl_mean = np.cumsum(mean_pnl_rl)
    # Propagate standard deviation of the mean for the cumulative sum
    cum_pnl_std = np.sqrt(np.cumsum(std_pnl_rl**2))
    
    time_steps = np.arange(len(cum_pnl_mean))
    
    axs[0].plot(time_steps, cum_pnl_mean, label='RL Agent (Mean)', color='blue', linewidth=2)
    axs[0].fill_between(time_steps, cum_pnl_mean - cum_pnl_std, cum_pnl_mean + cum_pnl_std, color='blue', alpha=0.2, label='RL Agent (Std Dev)')
    
    axs[0].plot(np.cumsum(pnl_no_hedge), label='No Hedge (h=0.00)', color='red', linestyle='--')
    axs[0].plot(np.cumsum(pnl_full_hedge), label='Full Hedge (h=1.00)', color='green', linestyle='-.')
    axs[0].plot(np.cumsum(pnl_ols_hedge), label='OLS Hedge (h=0.92)', color='orange', linestyle=':')
    axs[0].set_title(f'Average Cumulative PnL Comparison ({N_EVALUATIONS} Runs)', fontsize=16)
    axs[0].set_ylabel('Cumulative PnL', fontsize=12)
    axs[0].legend(fontsize=12)
    axs[0].grid(True)

    # Plot 2: Hedge Ratios
    time_steps_ratios = np.arange(len(mean_ratios_rl))
    axs[1].plot(time_steps_ratios, mean_ratios_rl, label='RL Agent (Mean Ratio)', color='purple', drawstyle='steps-post')
    axs[1].fill_between(time_steps_ratios, mean_ratios_rl - std_ratios_rl, mean_ratios_rl + std_ratios_rl, color='purple', alpha=0.2, step='post', label='RL Agent (Std Dev)')
    
    axs[1].axhline(y=1.0, color='green', linestyle='-.', label='Full Hedge (1.0)')
    axs[1].axhline(y=0.0, color='red', linestyle='--', label='No Hedge (0.0)')
    axs[1].axhline(y=0.92, color='orange', linestyle=':', label='OLS Hedge (0.92)')
    axs[1].set_title('Hedge Ratio Policies', fontsize=16)
    axs[1].set_ylabel('Hedge Ratio', fontsize=12)
    axs[1].set_xlabel('Time Step (Days)', fontsize=12)
    axs[1].set_ylim(-0.1, 1.6)
    axs[1].legend(fontsize=12)
    axs[1].grid(True)

    plt.tight_layout()
    timestamp = pd.Timestamp.now().strftime('%d%m%y_%H%M')
    output_path = f'v3/logs/v3_risk_reduction_comparison_avg_{timestamp}.png'
    plt.savefig(output_path)
    print(f"\nAverage comparison plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
