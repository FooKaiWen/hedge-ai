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

# Ignore warnings from deprecated packages
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    # --- Load Data ---
    data_path = os.path.join(base_path, 'data', 'palm_oil_enriched_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Drop NaNs from the entire dataframe to be consistent with training
    df.dropna(inplace=True)
    
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:].reset_index(drop=True)

    # --- Load Features, Model, and Normalization Stats ---
    model_path = os.path.join(base_path, 'models', 'ppo_hedge_v3.zip')
    vec_norm_path = os.path.join(base_path, 'models', 'vec_normalize_v3.pkl')
    features_path = os.path.join(base_path, 'models', 'features.json')

    if not all(os.path.exists(p) for p in [model_path, vec_norm_path, features_path]):
        raise FileNotFoundError("Trained model, normalization stats, or features file not found. Please run 'v3/main.py' to train and save these artifacts first.")

    # Load the exact feature list used during training
    with open(features_path, 'r') as f:
        features = json.load(f)
    print(f"Loaded {len(features)} features from v3/models/features.json")

    # The environment needs to be created with the correct features before loading the stats
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

def evaluate_rl_agent(model, env, df_test):
    """Evaluates the loaded RL agent on the test set."""
    obs = env.reset()
    pnl = []
    hedge_ratios = []
    
    # We can't use the environment's internal PnL calculation because it's stateful
    # and depends on a random start. We must evaluate over the whole test set.
    current_hedge_ratio = 0.0
    
    for i in range(len(df_test) - 1):
        # The observation must be constructed manually for a sequential evaluation
        market_features = df_test.iloc[i][env.envs[0].features].values.astype(np.float32)
        obs_manual = np.concatenate([market_features, [current_hedge_ratio]])
        
        # Normalize the manually created observation
        obs_normalized = env.normalize_obs(obs_manual)
        
        action, _ = model.predict(obs_normalized, deterministic=True)
        new_hedge_ratio = action[0]
        
        # Calculate PnL and cost for this step
        spot_return = df_test['spot_ret'].iloc[i]
        futures_return = df_test['fut_ret'].iloc[i]
        
        step_pnl = spot_return - current_hedge_ratio * futures_return
        cost = 0.0005 * abs(new_hedge_ratio - current_hedge_ratio)
        net_pnl = step_pnl - cost
        
        pnl.append(net_pnl)
        hedge_ratios.append(current_hedge_ratio)
        
        # Update hedge ratio for the next step
        current_hedge_ratio = new_hedge_ratio
        
    return np.array(pnl), np.array(hedge_ratios)

def calculate_benchmarks(df_train, df_test):
    """Calculates performance for No-Hedge, Full-Hedge, and OLS-Hedge strategies."""
    results = {}
    test_spot_ret = df_test['spot_ret'].iloc[:-1]
    test_fut_ret = df_test['fut_ret'].iloc[:-1]

    # --- 1. No Hedge (h=0) ---
    results['No Hedge'] = {'pnl': test_spot_ret.values, 'h': 0}

    # --- 2. Full Hedge (h=1) ---
    # Assuming no transaction costs for a static hedge
    pnl_full_hedge = test_spot_ret - 1.0 * test_fut_ret
    results['Full Hedge'] = {'pnl': pnl_full_hedge.values, 'h': 1}

    # --- 3. OLS / MVHR Hedge ---
    X_train = df_train[['fut_ret']].dropna()
    y_train = df_train['spot_ret'].loc[X_train.index]
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    h_ols = lr.coef_[0]
    
    pnl_ols_hedge = test_spot_ret - h_ols * test_fut_ret
    results['OLS Hedge'] = {'pnl': pnl_ols_hedge.values, 'h': h_ols}
    
    print(f"Calculated OLS (MVHR) hedge ratio: {h_ols:.4f}")
    return results

def calculate_metrics(pnl_series):
    """Calculates performance metrics for a series of PnL."""
    if len(pnl_series) == 0:
        return 0, 0, 0
    
    # Use 252 trading days for annualization
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

def plot_comparison(rl_pnl, rl_ratios, benchmarks, df_test):
    """Plots RL agent vs. benchmarks."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    
    # --- Plot 1: Cumulative PnL ---
    axs[0].plot(np.cumsum(rl_pnl), label='RL Agent', color='blue', linewidth=2.5, zorder=5)
    
    colors = {'No Hedge': 'red', 'Full Hedge': 'green', 'OLS Hedge': 'orange'}
    for name, data in benchmarks.items():
        axs[0].plot(np.cumsum(data['pnl']), label=f"{name} (h={data['h']:.2f})", color=colors[name], linestyle='--')

    axs[0].set_title('RL Agent vs. Benchmark Strategies: Cumulative PnL', fontsize=16)
    axs[0].set_ylabel('Cumulative PnL', fontsize=12)
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # --- Plot 2: Hedge Ratios ---
    axs[1].plot(df_test.index[:-1], rl_ratios, label='RL Agent Dynamic Ratio', color='purple', drawstyle='steps-post', zorder=5)
    axs[1].axhline(y=benchmarks['Full Hedge']['h'], color='green', linestyle='--', label=f"Full Hedge Ratio ({benchmarks['Full Hedge']['h']:.2f})")
    axs[1].axhline(y=benchmarks['OLS Hedge']['h'], color='orange', linestyle='--', label=f"OLS Hedge Ratio ({benchmarks['OLS Hedge']['h']:.2f})")
    
    axs[1].set_title('Hedge Ratio Policies', fontsize=16)
    axs[1].set_ylabel('Hedge Ratio', fontsize=12)
    axs[1].set_xlabel('Time Step (Days)', fontsize=12)
    axs[1].legend(loc='upper left')
    axs[1].grid(True)

    plt.tight_layout()
    save_path = 'v3/logs/v3_benchmark_comparison.png'
    plt.savefig(save_path)
    print(f"\nComparison plot saved to {save_path}")
    plt.show()

def main():
    """Main function to run the comparison."""
    try:
        # --- Load everything ---
        model, env, df_train, df_test, features = load_data_and_models()

        # --- Evaluate RL Agent ---
        print("\nEvaluating RL Agent...")
        rl_pnl, rl_ratios = evaluate_rl_agent(model, env, df_test)

        # --- Calculate Benchmarks ---
        print("\nCalculating benchmarks...")
        benchmarks = calculate_benchmarks(df_train, df_test)
        
        # --- Calculate and Print Metrics ---
        metrics = {}
        metrics['RL Agent'] = calculate_metrics(rl_pnl)
        for name, data in benchmarks.items():
            metrics[name] = calculate_metrics(data['pnl'])
            
        # Add VRE for RL Agent
        vre_rl = 1 - (metrics['RL Agent'][1] / metrics['No Hedge'][1]) if metrics['No Hedge'][1] > 0 else 0
        
        print("\n--- Performance Comparison ---")
        print(f"{'Strategy':<15} | {'Cumulative PnL':>18} | {'Variance':>12} | {'Sharpe Ratio (Ann.)':>22} | {'VRE':>8}")
        print("-" * 85)
        
        print(f"{'RL Agent':<15} | {metrics['RL Agent'][0]:>18.4f} | {metrics['RL Agent'][1]:>12.6f} | {metrics['RL Agent'][2]:>22.4f} | {vre_rl:>8.4f}")
        
        for name in benchmarks:
            vre = 1 - (metrics[name][1] / metrics['No Hedge'][1]) if metrics['No Hedge'][1] > 0 else 0
            print(f"{name:<15} | {metrics[name][0]:>18.4f} | {metrics[name][1]:>12.6f} | {metrics[name][2]:>22.4f} | {vre:>8.4f}")
        print("-" * 85)
        print("*VRE: Variance Reduction Effectiveness compared to No Hedge.")


        # --- Plot results ---
        plot_comparison(rl_pnl, rl_ratios, benchmarks, df_test)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
