import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN # CHANGED
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

# --- Data Loading and Preparation ---
def load_split_data(data_dir='data/rl_ready'):
    """
    Loads pre-split and preprocessed data for training, validation, and testing.
    """
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    df_train = pd.read_csv(train_path, parse_dates=['date'])
    df_val = pd.read_csv(val_path, parse_dates=['date'])
    df_test = pd.read_csv(test_path, parse_dates=['date'])

    # Feature engineering for environment
    for df in [df_train, df_val, df_test]:
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        # Lagged returns
        df['spot_ret_lag'] = df['spot_ret'].shift(1)
        df['fut_ret_lag'] = df['fut_ret'].shift(1)
        # Expected returns from predictions
        df['exp_day_ret'] = (df['next_day_pred'] - df['fut_close']) / df['fut_close']
        df['exp_month_ret'] = (df['next_month_pred'] - df['fut_close']) / df['fut_close']
        # Drop NaNs created by lagging
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df_train, df_val, df_test

# --- Hedging Environment for Discrete Actions ---
class HedgingEnv(gym.Env):
    def __init__(self, df):
        super(HedgingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        self.current_step = 0
        
        self.features = [
            'spot_ret_lag', 'fut_ret_lag', 'spot_std_20', 'fut_std_20',
            'basis', 'volume_z', 'hr_ols_60'
        ]
        
        # CHANGED: Discrete action space
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32)
        self.last_hedge_ratio = 0

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.hedged_returns = []
        return self._get_obs(), {}
    
    def step(self, action):
        # Handle action format
        if isinstance(action, np.ndarray):
            action = action.flatten()[0]
        
        # Map discrete action to hedge ratio
        h = float(action) * 0.2
        self.last_hedge_ratio = h

        spot_r = self.df['spot_ret'].iloc[self.current_step]
        fut_r = self.df['fut_ret'].iloc[self.current_step]
        
        hedged_r = spot_r - h * fut_r
        self.hedged_returns.append(hedged_r)
        
        reward = -(hedged_r ** 2)
        
        self.current_step += 1
        terminated = self.current_step > self.max_steps
        truncated = False
        
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        return self.df[self.features].iloc[self.current_step].values.astype(np.float32)

# --- Model Evaluation ---

def evaluate_rl_agent(model, df_test):
    """Evaluates the RL agent on the entire test set."""
    env = HedgingEnv(df_test)
    obs, info = env.reset()
    
    actions = []
    returns = []
    terminated = False
    
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        actions.append(env.last_hedge_ratio)
        if not terminated:
            returns.append(env.hedged_returns[-1])

    return np.array(returns), np.array(actions)

def evaluate_ols_static(df_train, df_test):
    """Computes hedge ratio from training data and applies it to test data."""
    X_train = df_train[['fut_ret']].dropna()
    y_train = df_train['spot_ret'].loc[X_train.index]
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    h_ols = lr.coef_[0]
    
    spot_rets = df_test['spot_ret']
    fut_rets = df_test['fut_ret']
    
    returns = spot_rets - h_ols * fut_rets
    return returns.values, np.full_like(returns, h_ols)

def evaluate_ols_rolling(df_train, df_test, window_size=60):
    """Computes rolling hedge ratio and applies it to test data."""
    full_df = pd.concat([df_train, df_test]).reset_index(drop=True)
    
    hedge_ratios = []
    returns = []
    
    test_start_index = len(df_train)
    
    for i in range(test_start_index, len(full_df)):
        window_start = i - window_size
        window_end = i
        
        window_df = full_df.iloc[window_start:window_end]
        
        X_window = window_df[['fut_ret']]
        y_window = window_df['spot_ret']
        
        lr = LinearRegression()
        lr.fit(X_window, y_window)
        h_rolling = lr.coef_[0]
        hedge_ratios.append(h_rolling)
        
        spot_r = full_df['spot_ret'].iloc[i]
        fut_r = full_df['fut_ret'].iloc[i]
        
        hedged_r = spot_r - h_rolling * fut_r
        returns.append(hedged_r)
        
    return np.array(returns), np.array(hedge_ratios)

# --- P&L Calculation and Plotting ---

def calculate_pnl_metrics(returns, label):
    """Calculates and prints P&L metrics."""
    total_pnl = np.sum(returns)
    variance = np.var(returns)
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
    
    print(f"--- {label} ---")
    print(f"  Cumulative P&L: {total_pnl:,.2f}")
    print(f"  Variance (Risk): {variance:.6f}")
    print(f"  Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return {'pnl': total_pnl, 'var': variance, 'sharpe': sharpe_ratio}

def plot_pnl_comparison(results):
    """Plots cumulative P&L and hedge ratios."""
    fig, axs = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Plot Cumulative P&L
    for label, data in results.items():
        axs[0].plot(np.cumsum(data['returns']), label=f"{label} (P&L: {data['metrics']['pnl']:,.0f})")
    
    axs[0].set_title('Out-of-Sample Cumulative P&L Comparison')
    axs[0].set_ylabel('Cumulative P&L')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot Hedge Ratios
    for label, data in results.items():
        if 'actions' in data:
            axs[1].plot(data['actions'], label=f'{label} Hedge Ratio', alpha=0.8)
            
    axs[1].set_title('Hedge Ratios Over Time')
    axs[1].set_xlabel('Time Steps (Days)')
    axs[1].set_ylabel('Hedge Ratio')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('logs/pnl_comparison_dqn.png') # CHANGED
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    
    # Load data
    df_train, df_val, df_test = load_split_data(data_dir='data/rl_ready')
    
    # --- Evaluate Models ---
    print("Evaluating models on the test set...")
    
    # 1. DQN Agent
    dqn_model_path = 'models/dqn_hedging_model.zip' # CHANGED
    if os.path.exists(dqn_model_path):
        dqn_model = DQN.load(dqn_model_path) # CHANGED
        dqn_returns, dqn_actions = evaluate_rl_agent(dqn_model, df_test) # CHANGED
    else:
        print(f"DQN model not found at {dqn_model_path}. Skipping DQN evaluation.")
        dqn_returns, dqn_actions = np.array([]), np.array([])

    # 2. Static OLS (MVHR)
    ols_static_returns, ols_static_actions = evaluate_ols_static(df_train, df_test)
    
    # 3. Rolling OLS
    rolling_window = 60
    ols_rolling_returns, ols_rolling_actions = evaluate_ols_rolling(df_train, df_test, window_size=rolling_window)
    
    # 4. Unhedged
    unhedged_returns = df_test['spot_ret'].values
    
    # --- Calculate and Print P&L Metrics ---
    print("\n--- Out-of-Sample P&L Performance ---")
    
    results = {}
    if dqn_returns.any():
        results['DQN Agent'] = { # CHANGED
            'returns': dqn_returns,
            'actions': dqn_actions,
            'metrics': calculate_pnl_metrics(dqn_returns, 'DQN Agent') # CHANGED
        }
    
    results['Static OLS'] = {
        'returns': ols_static_returns,
        'actions': ols_static_actions,
        'metrics': calculate_pnl_metrics(ols_static_returns, 'Static OLS')
    }
    
    results[f'Rolling OLS ({rolling_window}d)'] = {
        'returns': ols_rolling_returns,
        'actions': ols_rolling_actions,
        'metrics': calculate_pnl_metrics(ols_rolling_returns, f'Rolling OLS ({rolling_window}d)')
    }
    
    results['Unhedged'] = {
        'returns': unhedged_returns,
        'metrics': calculate_pnl_metrics(unhedged_returns, 'Unhedged')
    }
    
    # --- Plot Results ---
    plot_pnl_comparison(results)
