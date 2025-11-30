import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import torch as th

# ==============================================================================
# 1. ENVIRONMENT DEFINITION (Copied from main_cost_control.py for consistency)
# ==============================================================================
class HedgeEnvEval(gym.Env):
    """
    A generic evaluation environment. The reward function is not used during
    evaluation, so we only need the core mechanics (step, obs) to be consistent.
    """
    def __init__(self, df, features, episode_length=90, transaction_cost=0.0005):
        super(HedgeEnvEval, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.features = features
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.max_steps = len(df) - episode_length - 1

        self.action_space = spaces.Box(low=0.0, high=1.5, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features) + 1,), dtype=np.float32
        )
        
        self.current_hedge_ratio = 0.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        
        # For evaluation, always start from the beginning of the test set
        self.start_step = 0
        self.current_step = 0
        
        self.current_hedge_ratio = 0.0
        self.total_pnl = 0
        
        return self._get_obs(), {}

    def step(self, action):
        new_hedge_ratio = action[0]
        
        abs_step = self.start_step + self.current_step
        
        spot_return = self.df['spot_ret'].iloc[abs_step]
        futures_return = self.df['fut_ret'].iloc[abs_step]
        pnl = spot_return - self.current_hedge_ratio * futures_return
        
        cost = self.transaction_cost * abs(new_hedge_ratio - self.current_hedge_ratio)
        
        # Reward is irrelevant for evaluation, but we calculate it for completeness
        reward = pnl - cost
        
        self.current_hedge_ratio = new_hedge_ratio
        self.total_pnl += pnl
        self.current_step += 1
        
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        obs = self._get_obs()
        info = {'pnl': pnl, 'cost': cost, 'hedge_ratio': self.current_hedge_ratio}
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        abs_step = self.start_step + self.current_step
        if abs_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        market_features = self.df[self.features].iloc[abs_step].values.astype(np.float32)
        obs = np.concatenate([market_features, [self.current_hedge_ratio]])
        return obs

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================
def load_full_dataset(base_path='v3/data'):
    full_df_path = os.path.join(base_path, 'palm_oil_enriched_data.csv')
    if not os.path.exists(full_df_path):
        raise FileNotFoundError(f"Ensure '{full_df_path}' exists.")
        
    df = pd.read_csv(full_df_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    return df

# ==============================================================================
# 3. EVALUATION FUNCTIONS
# ==============================================================================
def evaluate_agent(df_test, model_path, vec_norm_path, features_path, transaction_cost=0.0005):
    """Loads a trained agent and evaluates its performance on the test set."""
    if not all(os.path.exists(p) for p in [model_path, vec_norm_path, features_path]):
        print(f"Skipping evaluation for {model_path} - one or more files are missing.")
        return None

    with open(features_path, 'r') as f:
        features = json.load(f)

    # Ensure all features are in the dataframe
    missing_feats = [f for f in features if f not in df_test.columns]
    if missing_feats:
        print(f"Skipping {model_path}: Missing features in test data: {missing_feats}")
        return None

    eval_env_raw = DummyVecEnv([lambda: HedgeEnvEval(df_test, features=features, transaction_cost=transaction_cost)])
    
    # Load the saved statistics object to get the running moving averages
    saved_stats = VecNormalize.load(vec_norm_path, eval_env_raw)
    
    # Create a fresh VecNormalize wrapper around the evaluation environment
    env = VecNormalize(eval_env_raw, training=False, norm_reward=False)
    
    # Manually copy the observation running mean and variance from the loaded stats
    env.obs_rms = saved_stats.obs_rms


    model = PPO.load(model_path, env=env)

    obs = env.reset()
    pnls, costs, hedge_ratios, entropies = [], [], [], []
    
    # Loop for the length of one episode on the test set
    for _ in range(len(df_test) - 91):
        obs_tensor = th.as_tensor(obs, device=model.device)
        distribution = model.policy.get_distribution(obs_tensor)
        action = distribution.mode()  # Use mode for deterministic action (mean)
        entropy = distribution.entropy()

        # Clip the action to the valid range before executing
        clipped_action = th.clamp(action, env.action_space.low[0], env.action_space.high[0])

        obs, rewards, dones, infos = env.step(clipped_action.detach().cpu().numpy())
        
        # Extract info from the first (and only) env
        info = infos[0]
        
        pnls.append(info['pnl'])
        costs.append(info['cost'])
        hedge_ratios.append(info['hedge_ratio'])
        entropies.append(entropy.detach().cpu().numpy().flatten()[0])
    
    return {'pnl': pnls, 'costs': costs, 'hedge_ratios': hedge_ratios, 'entropies': entropies}

def evaluate_static_hedge(df_test, static_ratio=1.0, transaction_cost=0.0005):
    """Evaluates a static hedging strategy over the same period as the agent."""
    eval_period_len = len(df_test) - 91  # Match agent eval length

    spot_returns = df_test['spot_ret'].iloc[:eval_period_len]
    futures_returns = df_test['fut_ret'].iloc[:eval_period_len]
    
    # Cost is incurred once at the start to establish the position
    initial_cost = transaction_cost * static_ratio
    costs = [initial_cost] + [0] * (len(spot_returns) - 1)
    
    pnls = spot_returns - static_ratio * futures_returns
    hedge_ratios = [static_ratio] * len(pnls)
    entropies = [0] * len(pnls)  # A deterministic policy has zero entropy
    
    return {'pnl': pnls.tolist(), 'costs': costs, 'hedge_ratios': hedge_ratios, 'entropies': entropies}

def evaluate_no_hedge(df_test):
    """Evaluates a 'no hedge' strategy over the same period as the agent."""
    eval_period_len = len(df_test) - 91  # Match agent eval length
    pnls = df_test['spot_ret'].iloc[:eval_period_len]
    costs = [0] * len(pnls)
    hedge_ratios = [0] * len(pnls)
    entropies = [0] * len(pnls)  # A deterministic policy has zero entropy
    
    return {'pnl': pnls.tolist(), 'costs': costs, 'hedge_ratios': hedge_ratios, 'entropies': entropies}


def calculate_metrics(results):
    """Calculates summary statistics from evaluation results."""
    pnl = np.array(results['pnl'])
    costs = np.array(results['costs'])
    hedge_ratios = np.array(results['hedge_ratios'])
    
    total_pnl = pnl.sum()
    total_cost = costs.sum()
    net_pnl = total_pnl - total_cost
    
    # Sharpe Ratio (annualized, assuming daily returns)
    sharpe_ratio = (pnl.mean() / (pnl.std() + 1e-9)) * np.sqrt(252)
    
    # Portfolio Turnover
    turnover = np.sum(np.abs(np.diff(hedge_ratios)))
    
    return {
        'Total PnL': total_pnl,
        'Net PnL': net_pnl,
        'Total Costs': total_cost,
        'Sharpe Ratio': sharpe_ratio,
        'Turnover': turnover
    }

# ==============================================================================
# 4. MAIN COMPARISON SCRIPT
# ==============================================================================
def main():
    print("--- Starting Agent Comparison ---")
    
    # --- Load Data ---
    full_df = load_full_dataset()
    _ , df_test = np.split(full_df, [int(len(full_df) * 0.8)])
    
    # --- Define Models & Baselines ---
    agents_to_compare = {
        "Cost Control": {
            "model_path": "v3/models/ppo_hedge_cost_control_20251126_221942.zip",
            "vec_norm_path": "v3/models/vec_normalize_cost_control_20251126_221942.pkl",
            "features_path": "v3/models/features_cost_control_20251126_221942.json"
        },
        "Profit Maximization": {
            "model_path": "v3/models/ppo_hedge_profit_maximization_20251118_223523.zip",
            "vec_norm_path": "v3/models/vec_normalize_profit_maximization_20251118_223523.pkl",
            "features_path": "v3/models/features_profit_maximization_20251118_223523.json"
        },
        "Risk Reduction": {
            "model_path": "v3/models/ppo_hedge_risk_reduction_20251118_223523.zip",
            "vec_norm_path": "v3/models/vec_normalize_risk_reduction_20251118_223523.pkl",
            "features_path": "v3/models/features_risk_reduction_20251118_223523.json"
        }
    }

    # --- Run Evaluation ---
    all_results = {}
    for name, paths in agents_to_compare.items():
        print(f"Evaluating {name} agent...")
        results = evaluate_agent(df_test.copy(), **paths)
        if results:
            all_results[name] = results
            
    print("Evaluating Static Buy-and-Hold baseline...")
    all_results['Static (Ratio=1.0)'] = evaluate_static_hedge(df_test.copy(), static_ratio=1.0)
    
    print("Evaluating No-Hedge baseline...")
    all_results['No Hedge (Ratio=0.0)'] = evaluate_no_hedge(df_test.copy())
    
    # --- Calculate & Print Metrics ---
    summary_metrics = {}
    print("\n--- Comparative Performance Metrics ---")
    for name, results in all_results.items():
        metrics = calculate_metrics(results)
        summary_metrics[name] = metrics
        print(f"\n--- {name} ---")
        for key, value in metrics.items():
            print(f"{key:<15}: {value: .5f}")

    # --- Plotting ---
    print("\n--- Generating Comparison Plots ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(4, 1, figsize=(18, 32), sharex=True)
    colors = {'Cost Control': 'red', 'Profit Maximization': 'blue', 'Risk Reduction': 'green', 'Static (Ratio=1.0)': 'black', 'No Hedge (Ratio=0.0)': 'orange'}

    # Plot 1: Cumulative Net PnL
    for name, results in all_results.items():
        net_pnl = np.cumsum(np.array(results['pnl']) - np.array(results['costs']))
        axs[0].plot(net_pnl, label=name, color=colors.get(name, 'gray'))
    axs[0].set_title('Comparative Net PnL (PnL - Costs)', fontsize=16)
    axs[0].set_ylabel('Cumulative Net PnL', fontsize=12)
    axs[0].legend()
    axs[0].axhline(0, color='black', linestyle='--', linewidth=0.8)

    # Plot 2: Cumulative Costs
    for name, results in all_results.items():
        axs[1].plot(np.cumsum(results['costs']), label=name, color=colors.get(name, 'gray'))
    axs[1].set_title('Comparative Cumulative Transaction Costs', fontsize=16)
    axs[1].set_ylabel('Cumulative Costs', fontsize=12)
    axs[1].legend()

    # Plot 3: Hedge Ratio Policies
    for name, results in all_results.items():
        axs[2].plot(results['hedge_ratios'], label=name, color=colors.get(name, 'gray'), alpha=0.8, drawstyle='steps-post')
    axs[2].set_title('Comparative Hedge Ratio Policies', fontsize=16)
    axs[2].set_ylabel('Hedge Ratio', fontsize=12)
    axs[2].legend()

    # Plot 4: Policy Uncertainty (Entropy)
    for name, results in all_results.items():
        if 'entropies' in results: # Only plot for agents that have uncertainty
            axs[3].plot(results['entropies'], label=name, color=colors.get(name, 'gray'), alpha=0.8)
    axs[3].set_title('Comparative Policy Uncertainty (Entropy)', fontsize=16)
    axs[3].set_ylabel('Entropy', fontsize=12)
    axs[3].set_xlabel('Time Step in Test Set', fontsize=12)
    axs[3].legend()

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'v3/logs/agent_comparison_results_{timestamp}.png'
    plt.savefig(save_path)
    print(f"\n--- Comparison complete. Plot saved to {save_path} ---")
    plt.show()

if __name__ == "__main__":
    main()
