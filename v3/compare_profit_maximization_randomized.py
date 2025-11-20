import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sklearn.linear_model import LinearRegression
import warnings
import json
import gymnasium as gym
from tqdm import tqdm

# Assuming main.py contains the correct environment definition
from main_profit_maximization import HedgeEnvV3, load_full_dataset

# Ignore warnings from deprecated packages
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Number of times to run the evaluation to get a stable average
N_EVALUATIONS = 50

# ==============================================================================
# 1. AGENT & BENCHMARK EVALUATION
# ==============================================================================

def calculate_metrics(pnl_series):
    """Calculates performance metrics for a series of PnL."""
    if pnl_series is None or len(pnl_series) == 0:
        return 0, 0, 0
    
    annual_factor = 252 / np.sqrt(252)
    cumulative_pnl = np.sum(pnl_series)
    variance = np.var(pnl_series)
    mean_return = np.mean(pnl_series)
    std_return = np.std(pnl_series)
    sharpe_ratio = (mean_return / std_return) * annual_factor if std_return > 0 else 0
    
    return cumulative_pnl, variance, sharpe_ratio

def evaluate_strategy(df_test, strategy, model=None, vec_normalize_path=None, features=None, seed=None, ols_hedge_ratio=None):
    """Evaluates a given hedging strategy on the test set."""
    pnl_history = []
    hedge_ratios = []

    if strategy == 'rl':
        # --- RL Agent Evaluation with random start ---
        env_raw = DummyVecEnv([lambda: HedgeEnvV3(df_test, features=features)])
        if seed is not None:
            env_raw.seed(seed)
        
        env = VecNormalize.load(vec_normalize_path, env_raw)
        env.training = False
        env.norm_reward = False
        
        obs = env.reset()
        # Run for a long episode, starting from a random point
        for _ in range(len(df_test) - 91):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, info = env.step(action)
            # The 'pnl' must be in the info dict, as per the environment's design
            pnl_history.append(info[0]['pnl'])
            hedge_ratios.append(env.envs[0].current_hedge_ratio)
    
    else: # Benchmark strategies
        # --- Benchmarks are calculated over the full test set for consistency ---
        spot_returns = df_test['spot_ret'].iloc[:-1].values
        futures_returns = df_test['fut_ret'].iloc[:-1].values
        
        current_hedge = 0
        if strategy == 'full':
            current_hedge = 1.0
        elif strategy == 'ols':
            current_hedge = ols_hedge_ratio if ols_hedge_ratio is not None else 0.9 # Fallback

        pnl_history = spot_returns - current_hedge * futures_returns
        hedge_ratios = [current_hedge] * len(pnl_history)

    return pnl_history, hedge_ratios

# ==============================================================================
# 2. PLOTTING & REPORTING
# ==============================================================================

def plot_comparison(rl_pnl_mean, rl_pnl_std, rl_ratios_mean, rl_ratios_std, benchmarks, df_test):
    """Plots RL agent vs. benchmarks."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    time_steps = np.arange(len(rl_pnl_mean))

    # --- Plot 1: Cumulative PnL ---
    cum_pnl_mean = np.cumsum(rl_pnl_mean)
    cum_pnl_std = np.sqrt(np.cumsum(rl_pnl_std**2))

    axs[0].plot(time_steps, cum_pnl_mean, label='RL Agent (Mean)', color='blue', linewidth=2.5, zorder=5)
    axs[0].fill_between(time_steps, cum_pnl_mean - cum_pnl_std, cum_pnl_mean + cum_pnl_std, color='blue', alpha=0.2, label='RL Agent (Std Dev)')

    colors = {'No Hedge': 'red', 'Full Hedge': 'green', 'OLS Hedge': 'orange'}
    for name, data in benchmarks.items():
        # Ensure benchmark PnL is same length as RL PnL for plotting
        pnl_to_plot = data['pnl'][:len(time_steps)]
        axs[0].plot(np.cumsum(pnl_to_plot), label=f"{name} (h={data['h']:.2f})", color=colors[name], linestyle='--')

    axs[0].set_title(f'RL Agent vs. Benchmarks: Average Cumulative PnL ({N_EVALUATIONS} Runs)', fontsize=16)
    axs[0].set_ylabel('Cumulative PnL', fontsize=12)
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # --- Plot 2: Hedge Ratios ---
    axs[1].plot(time_steps, rl_ratios_mean, label='RL Agent (Mean Ratio)', color='purple', drawstyle='steps-post', zorder=5)
    axs[1].fill_between(time_steps, rl_ratios_mean - rl_ratios_std, rl_ratios_mean + rl_ratios_std, color='purple', alpha=0.2, step='post', label='RL Agent (Std Dev)')
    
    axs[1].axhline(y=benchmarks['Full Hedge']['h'], color='green', linestyle='--', label=f"Full Hedge Ratio ({benchmarks['Full Hedge']['h']:.2f})")
    axs[1].axhline(y=benchmarks['OLS Hedge']['h'], color='orange', linestyle='--', label=f"OLS Hedge Ratio ({benchmarks['OLS Hedge']['h']:.2f})")
    
    axs[1].set_title('Hedge Ratio Policies', fontsize=16)
    axs[1].set_ylabel('Hedge Ratio', fontsize=12)
    axs[1].set_xlabel('Time Step (Days)', fontsize=12)
    axs[1].legend(loc='upper left')
    axs[1].grid(True)

    plt.tight_layout()
    timestamp = pd.Timestamp.now().strftime('%d%m%y_%H%M')
    save_path = f'v3/logs/v3_profit_maximization_randomized_avg_{timestamp}.png'
    plt.savefig(save_path)
    print(f"\nComparison plot saved to {save_path}")
    plt.show()

def main():
    """Main function to run the comparison."""
    try:
        # --- Load Data & Models ---
        base_path = 'v3'
        df_full = load_full_dataset(base_path=os.path.join(base_path, 'data'))
        train_size = int(len(df_full) * 0.8)
        df_train = df_full.iloc[:train_size]
        df_test = df_full.iloc[train_size:].reset_index(drop=True)

        # Using the specific model paths for the profit maximization agent
        model_path = os.path.join(base_path, 'models', 'ppo_hedge_profit_maximization_20251118_223523.zip')
        vec_norm_path = os.path.join(base_path, 'models', 'vec_normalize_profit_maximization_20251118_223523.pkl')
        features_path = os.path.join(base_path, 'models', 'features_profit_maximization_20251118_223523.json')

        model = PPO.load(model_path)
        with open(features_path, 'r') as f:
            features = json.load(f)
        
        # --- Evaluate RL Agent (Monte Carlo) ---
        all_pnl_rl, all_ratios_rl = [], []
        print(f"\nEvaluating Profit-Maximization RL Agent over {N_EVALUATIONS} runs...")
        for i in tqdm(range(N_EVALUATIONS)):
            pnl, ratios = evaluate_strategy(df_test, 'rl', model, vec_norm_path, features, seed=i)
            all_pnl_rl.append(pnl)
            all_ratios_rl.append(ratios)
        
        rl_pnl_mean = np.mean(all_pnl_rl, axis=0)
        rl_pnl_std = np.std(all_pnl_rl, axis=0)
        rl_ratios_mean = np.mean(all_ratios_rl, axis=0)
        rl_ratios_std = np.std(all_ratios_rl, axis=0)

        # --- Calculate Benchmarks ---
        X_train = df_train[['fut_ret']].dropna()
        y_train = df_train['spot_ret'].loc[X_train.index]
        lr = LinearRegression().fit(X_train, y_train)
        h_ols = lr.coef_[0]
        print(f"Calculated OLS (MVHR) hedge ratio: {h_ols:.4f}")

        benchmarks = {}
        pnl_no_hedge, _ = evaluate_strategy(df_test, 'none')
        pnl_full_hedge, _ = evaluate_strategy(df_test, 'full')
        pnl_ols_hedge, _ = evaluate_strategy(df_test, 'ols', ols_hedge_ratio=h_ols)
        benchmarks['No Hedge'] = {'pnl': pnl_no_hedge, 'h': 0}
        benchmarks['Full Hedge'] = {'pnl': pnl_full_hedge, 'h': 1}
        benchmarks['OLS Hedge'] = {'pnl': pnl_ols_hedge, 'h': h_ols}

        # --- Calculate and Print Metrics ---
        metrics = {}
        metrics['RL Agent (Avg)'] = calculate_metrics(rl_pnl_mean)
        for name, data in benchmarks.items():
            metrics[name] = calculate_metrics(data['pnl'])
            
        vre_rl = 1 - (metrics['RL Agent (Avg)'][1] / metrics['No Hedge'][1]) if metrics['No Hedge'][1] > 0 else 0
        
        print("\n--- Average Performance Comparison ---")
        print(f"{'Strategy':<15} | {'Cumulative PnL':>18} | {'Variance':>12} | {'Sharpe Ratio (Ann.)':>22} | {'VRE':>8}")
        print("-" * 85)
        print(f"{'RL Agent (Avg)':<15} | {metrics['RL Agent (Avg)'][0]:>18.4f} | {metrics['RL Agent (Avg)'][1]:>12.6f} | {metrics['RL Agent (Avg)'][2]:>22.4f} | {vre_rl:>8.4f}")
        for name in benchmarks:
            vre = 1 - (metrics[name][1] / metrics['No Hedge'][1]) if metrics['No Hedge'][1] > 0 else 0
            print(f"{name:<15} | {metrics[name][0]:>18.4f} | {metrics[name][1]:>12.6f} | {metrics[name][2]:>22.4f} | {vre:>8.4f}")
        print("-" * 85)
        print("*VRE: Variance Reduction Effectiveness compared to No Hedge.")

        plot_comparison(rl_pnl_mean, rl_pnl_std, rl_ratios_mean, rl_ratios_std, benchmarks, df_test)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
