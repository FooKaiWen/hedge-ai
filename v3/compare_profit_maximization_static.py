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
from gymnasium import spaces
from tqdm import tqdm

# Ignore warnings from deprecated packages
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Number of times to run the evaluation
N_EVALUATIONS = 50

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================

# Define a dummy environment class for loading the model
class HedgeEnvV3(gym.Env):
    def __init__(self, df, features, episode_length=90, transaction_cost=0.0005):
        super(HedgeEnvV3, self).__init__()
        self.df = df
        self.features = features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(features) + 1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.5, shape=(1,), dtype=np.float32)
    def reset(self, *, seed=None, options=None): return np.zeros(self.observation_space.shape), {}
    def step(self, action): return np.zeros(self.observation_space.shape), 0, False, False, {}

def load_data_and_models(base_path='v3'):
    """Loads the dataset, trained PPO model, and normalization stats."""
    data_path = os.path.join(base_path, 'data', 'palm_oil_enriched_data.csv')
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:].reset_index(drop=True)

    model_path = os.path.join(base_path, 'models', 'ppo_hedge_profit_maximization_20251118_223523.zip')
    vec_norm_path = os.path.join(base_path, 'models', 'vec_normalize_profit_maximization_20251118_223523.pkl')
    features_path = os.path.join(base_path, 'models', 'features_profit_maximization_20251118_223523.json')

    with open(features_path, 'r') as f:
        features = json.load(f)
    
    eval_env_raw = DummyVecEnv([lambda: HedgeEnvV3(df_test, features=features)])
    eval_env = VecNormalize.load(vec_norm_path, eval_env_raw)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env)
    
    print("Successfully loaded data, model, and normalization statistics.")
    return model, eval_env, df_train, df_test, features

# ==============================================================================
# 2. AGENT & BENCHMARK EVALUATION
# ==============================================================================

def evaluate_rl_agent(model, env, df_test, seed=None):
    """Evaluates the loaded RL agent on the test set. The seed is unused here as the evaluation is deterministic."""
    pnl = []
    hedge_ratios = []
    current_hedge_ratio = 0.0
    
    for i in range(len(df_test) - 1):
        market_features = df_test.iloc[i][env.envs[0].features].values.astype(np.float32)
        obs_manual = np.concatenate([market_features, [current_hedge_ratio]])
        obs_normalized = env.normalize_obs(obs_manual)
        
        action, _ = model.predict(obs_normalized, deterministic=True)
        new_hedge_ratio = action[0]
        
        spot_return = df_test['spot_ret'].iloc[i]
        futures_return = df_test['fut_ret'].iloc[i]
        
        step_pnl = spot_return - current_hedge_ratio * futures_return
        cost = 0.0005 * abs(new_hedge_ratio - current_hedge_ratio)
        net_pnl = step_pnl - cost
        
        pnl.append(net_pnl)
        hedge_ratios.append(current_hedge_ratio)
        current_hedge_ratio = new_hedge_ratio
        
    return np.array(pnl), np.array(hedge_ratios)

def calculate_benchmarks(df_train, df_test):
    """Calculates performance for No-Hedge, Full-Hedge, and OLS-Hedge strategies."""
    results = {}
    test_spot_ret = df_test['spot_ret'].iloc[:-1]
    test_fut_ret = df_test['fut_ret'].iloc[:-1]

    results['No Hedge'] = {'pnl': test_spot_ret.values, 'h': 0}
    results['Full Hedge'] = {'pnl': (test_spot_ret - 1.0 * test_fut_ret).values, 'h': 1}

    X_train = df_train[['fut_ret']].dropna()
    y_train = df_train['spot_ret'].loc[X_train.index]
    lr = LinearRegression().fit(X_train, y_train)
    h_ols = lr.coef_[0]
    
    results['OLS Hedge'] = {'pnl': (test_spot_ret - h_ols * test_fut_ret).values, 'h': h_ols}
    print(f"Calculated OLS (MVHR) hedge ratio: {h_ols:.4f}")
    return results

def calculate_metrics(pnl_series):
    """Calculates performance metrics for a series of PnL."""
    if len(pnl_series) == 0: return 0, 0, 0
    annual_factor = 252 / np.sqrt(252)
    cumulative_pnl = np.sum(pnl_series)
    variance = np.var(pnl_series)
    mean_return = np.mean(pnl_series)
    std_return = np.std(pnl_series)
    sharpe_ratio = (mean_return / std_return) * annual_factor if std_return > 0 else 0
    return cumulative_pnl, variance, sharpe_ratio

# ==============================================================================
# 3. PLOTTING & REPORTING
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
        axs[0].plot(np.cumsum(data['pnl']), label=f"{name} (h={data['h']:.2f})", color=colors[name], linestyle='--')

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
    save_path = f'v3/logs/v3_profit_maximization_comparison_avg_{timestamp}.png'
    plt.savefig(save_path)
    print(f"\nComparison plot saved to {save_path}")
    plt.show()

def main():
    """Main function to run the comparison."""
    try:
        model, env, df_train, df_test, features = load_data_and_models()

        # --- Evaluate RL Agent (Monte Carlo) ---
        all_pnl_rl, all_ratios_rl = [], []
        print(f"\nEvaluating RL Agent over {N_EVALUATIONS} runs...")
        for i in tqdm(range(N_EVALUATIONS)):
            pnl, ratios = evaluate_rl_agent(model, env, df_test, seed=i)
            all_pnl_rl.append(pnl)
            all_ratios_rl.append(ratios)
        
        rl_pnl_mean = np.mean(all_pnl_rl, axis=0)
        rl_pnl_std = np.std(all_pnl_rl, axis=0)
        rl_ratios_mean = np.mean(all_ratios_rl, axis=0)
        rl_ratios_std = np.std(all_ratios_rl, axis=0)

        benchmarks = calculate_benchmarks(df_train, df_test)
        
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
