import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Step 1: Data Preparation
# Assume you have a CSV file 'palm_oil_data.csv' with columns:
# - date (datetime)
# - spot_close (CPO spot price, daily - download historical from https://www.mpoc.org.my/market-insight/daily-palm-oil-prices/ or similar)
# - fut_close (FCPO close price)
# - volume (FCPO volume)
# - next_day_pred (predicted next day FCPO close)
# - next_month_pred (predicted next month FCPO close)
# - production (monthly CPO production, interpolated to daily)
# - oer (monthly OER rate, interpolated to daily)
# - ffb_price (daily FFB price, NaN where not available)

def load_and_preprocess_data(file_path='palm_oil_data.csv'):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Interpolate monthly data to daily
    df['production'] = df['production'].interpolate(method='linear').ffill().bfill()
    df['oer'] = df['oer'].interpolate(method='linear').ffill().bfill()
    
    # Handle FFB price (limited period, fill NaN with mean)
    df['ffb_price'] = df['ffb_price'].fillna(df['ffb_price'].mean())
    
    # Compute returns
    df['spot_ret'] = np.log(df['spot_close'] / df['spot_close'].shift(1))
    df['fut_ret'] = np.log(df['fut_close'] / df['fut_close'].shift(1))
    
    # Lagged returns
    df['spot_ret_lag'] = df['spot_ret'].shift(1)
    df['fut_ret_lag'] = df['fut_ret'].shift(1)
    
    # Rolling volatilities (20-day)
    df['spot_vol'] = df['spot_ret'].rolling(window=20).std()
    df['fut_vol'] = df['fut_ret'].rolling(window=20).std()
    
    # Expected returns from predictions
    df['exp_day_ret'] = (df['next_day_pred'] - df['fut_close']) / df['fut_close']
    df['exp_month_ret'] = (df['next_month_pred'] - df['fut_close']) / df['fut_close']
    
    # Normalize features
    scaler = MinMaxScaler()
    features_to_normalize = ['production', 'oer', 'ffb_price', 'spot_vol', 'fut_vol', 'volume']
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    
    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)
    
    return df

# Step 2: Define Custom Gymnasium Environment for RL Hedging
class HedgingEnv(gym.Env):
    def __init__(self, df, episode_length=30):
        super(HedgingEnv, self).__init__()
        self.df = df
        self.episode_length = episode_length
        self.max_steps = len(df) - episode_length - 1
        self.features = [
            'spot_ret_lag', 'fut_ret_lag', 'spot_vol', 'fut_vol',
            'production', 'oer', 'exp_day_ret', 'exp_month_ret',
            'ffb_price', 'volume'  # Add more if needed
        ]
        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)  # Hedge ratio between 0 and 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32)
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.start_step = np.random.randint(0, self.max_steps)
        self.current_step = 0
        self.hedged_returns = []
        return self._get_obs(), {}
    
    def step(self, action):
        h = action[0]
        abs_step = self.start_step + self.current_step
        spot_r = self.df['spot_ret'].iloc[abs_step]
        fut_r = self.df['fut_ret'].iloc[abs_step]
        hedged_r = spot_r - h * fut_r
        self.hedged_returns.append(hedged_r)
        
        reward = 0.0
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.episode_length
        if truncated:
            reward = -np.var(self.hedged_returns)
        
        obs = self._get_obs() if not truncated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        abs_step = self.start_step + self.current_step
        return self.df[self.features].iloc[abs_step].values.astype(np.float32)

# Step 3: Agent Training
def train_agent(df_train, model_path='ppo_hedging_model.zip'):
    env_fn = lambda: HedgingEnv(df_train)
    env = DummyVecEnv([env_fn])
    
    # Use PPO for continuous action space
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1, 
        n_steps=2048, 
        batch_size=64, 
        n_epochs=10, 
        gamma=0.99, 
        gae_lambda=0.95, 
        ent_coef=0.01  # Encourage exploration
    )
    
    # Evaluation callback
    eval_env = DummyVecEnv([env_fn])
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=10000, n_eval_episodes=10, deterministic=True, render=False)
    
    model.learn(total_timesteps=100000, callback=eval_callback)  # Adjust timesteps as needed
    model.save(model_path)
    return model

# Step 4: Agent Evaluation with Benchmarks
def evaluate_agent(model, df_test):
    env = HedgingEnv(df_test)
    obs, info = env.reset()
    hedged_returns_rl = []
    actions = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        h = action[0]
        actions.append(h)
        # Collect hedged returns from env's list (since reward is terminal)
    hedged_returns_rl = env.hedged_returns
    
    var_hedged_rl = np.var(hedged_returns_rl)
    var_unhedged = np.var(df_test['spot_ret'].iloc[env.start_step:env.start_step + env.episode_length])
    effectiveness_rl = 1 - (var_hedged_rl / var_unhedged) if var_unhedged != 0 else 0
    
    return effectiveness_rl, var_hedged_rl, hedged_returns_rl, actions

def compute_benchmarks(df_train, df_test, episode_start, episode_length):
    # Fixed ratio hedging (h=1)
    spot_rets = df_test['spot_ret'].iloc[episode_start:episode_start + episode_length]
    fut_rets = df_test['fut_ret'].iloc[episode_start:episode_start + episode_length]
    hedged_fixed = spot_rets - 1 * fut_rets
    var_fixed = np.var(hedged_fixed)
    
    # OLS/MVHR (static hedge ratio from train data)
    lr = LinearRegression()
    lr.fit(df_train[['fut_ret']], df_train['spot_ret'])
    h_ols = lr.coef_[0]
    hedged_ols = spot_rets - h_ols * fut_rets
    var_ols = np.var(hedged_ols)
    
    var_unhedged = np.var(spot_rets)
    effectiveness_fixed = 1 - (var_fixed / var_unhedged) if var_unhedged != 0 else 0
    effectiveness_ols = 1 - (var_ols / var_unhedged) if var_unhedged != 0 else 0
    
    return {
        'fixed': {'effectiveness': effectiveness_fixed, 'var': var_fixed, 'h': 1},
        'ols_mvhr': {'effectiveness': effectiveness_ols, 'var': var_ols, 'h': h_ols}
    }

# Step 5: Visualization
def plot_results(hedged_rl, hedged_fixed, hedged_ols, actions):
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    
    axs[0].plot(hedged_rl, label='RL Hedged Returns')
    axs[0].set_title('RL Hedged Returns')
    axs[0].legend()
    
    axs[1].plot(hedged_fixed, label='Fixed Ratio Hedged Returns')
    axs[1].plot(hedged_ols, label='OLS/MVHR Hedged Returns')
    axs[1].set_title('Benchmark Hedged Returns')
    axs[1].legend()
    
    axs[2].plot(actions, label='RL Hedge Ratios')
    axs[2].set_title('Dynamic Hedge Ratios from RL Agent')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('hedging_results.png')
    plt.show()

# End-to-End Pipeline
if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Split into train/test (80/20)
    split_idx = int(0.8 * len(df))
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    # Train the agent
    model = train_agent(df_train)
    
    # Evaluate on a sample episode from test
    effectiveness_rl, var_rl, hedged_rl, actions = evaluate_agent(model, df_test)
    
    # For benchmarks, use the same episode start as in eval
    benchmarks = compute_benchmarks(df_train, df_test, 0, 30)  # Assuming episode start=0 for simplicity; adjust as needed
    
    # Print results
    print(f"RL Effectiveness: {effectiveness_rl:.4f}, Variance: {var_rl:.6f}")
    print(f"Fixed Ratio Effectiveness: {benchmarks['fixed']['effectiveness']:.4f}, Variance: {benchmarks['fixed']['var']:.6f}, h: {benchmarks['fixed']['h']}")
    print(f"OLS/MVHR Effectiveness: {benchmarks['ols_mvhr']['effectiveness']:.4f}, Variance: {benchmarks['ols_mvhr']['var']:.6f}, h: {benchmarks['ols_mvhr']['h']:.4f}")
    
    # Plot
    hedged_fixed = df_test['spot_ret'].iloc[0:30] - 1 * df_test['fut_ret'].iloc[0:30]
    hedged_ols = df_test['spot_ret'].iloc[0:30] - benchmarks['ols_mvhr']['h'] * df_test['fut_ret'].iloc[0:30]
    plot_results(hedged_rl, hedged_fixed, hedged_ols, actions)