import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO # CHANGED
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

# Step 1: Data Preparation
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

    return df_train, df_val, df_test

# Step 2: Define Custom Gymnasium Environment for RL Hedging
class HedgingEnv(gym.Env):
    def __init__(self, df, episode_length=30):
        super(HedgingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.episode_length = episode_length
        self.max_steps = len(df) - episode_length - 1
        
        self.features = [
            'spot_ret_lag', 'fut_ret_lag', 'spot_std_20', 'fut_std_20',
            'basis', 'volume_z', 'hr_ols_60'
        ]
        
        # CHANGED: Discrete action space with 11 bins
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32)
        self.last_hedge_ratio = 0
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.start_step = np.random.randint(0, self.max_steps)
        self.current_step = 0
        self.hedged_returns = []
        return self._get_obs(), {}
    
    def step(self, action):
        # Handle both vectorized and non-vectorized envs
        if isinstance(action, np.ndarray):
            action = action.flatten()[0]

        # CHANGED: Map discrete action to continuous hedge ratio
        h = float(action) * 0.2
        self.last_hedge_ratio = h
        
        abs_step = self.start_step + self.current_step
        
        spot_r = self.df['spot_ret'].iloc[abs_step]
        fut_r = self.df['fut_ret'].iloc[abs_step]
        
        hedged_r = spot_r - h * fut_r
        self.hedged_returns.append(hedged_r)
        
        # Dense reward: negative squared hedged return
        reward = -(hedged_r ** 2)
        
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        obs = self._get_obs() if not (terminated or truncated) else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        abs_step = self.start_step + self.current_step
        return self.df[self.features].iloc[abs_step].values.astype(np.float32)

# Step 3: Agent Training
def train_agent(df_train, df_val, model_path='models/ppo_hedging_model.zip'):
    env_fn = lambda: HedgingEnv(df_train)
    env = DummyVecEnv([env_fn])
    
    # Evaluation callback using the validation set
    eval_env_fn = lambda: HedgingEnv(df_val)
    eval_env = DummyVecEnv([eval_env_fn])
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', 
                                 log_path='./logs/', eval_freq=5000, 
                                 n_eval_episodes=10, deterministic=True, render=False)
    
    # CHANGED: Use PPO agent
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1
    )
    
    # CHANGED: Train for ~2000 episodes (2000 * 30 = 60000 steps)
    model.learn(total_timesteps=60000, callback=eval_callback)
    model.save(model_path)
    return model

# Step 4: Agent Evaluation with Benchmarks
def evaluate_agent(model, df_test):
    env = HedgingEnv(df_test)
    obs, info = env.reset()
    
    actions = []
    terminated, truncated = False, False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(env.last_hedge_ratio)
        
    hedged_returns_rl = env.hedged_returns
    var_hedged_rl = np.var(hedged_returns_rl)
    
    episode_slice = slice(env.start_step, env.start_step + env.episode_length)
    var_unhedged = np.var(df_test['spot_ret'].iloc[episode_slice])
    
    effectiveness_rl = 1 - (var_hedged_rl / var_unhedged) if var_unhedged != 0 else 0
    
    return effectiveness_rl, var_hedged_rl, hedged_returns_rl, actions, env.start_step

def compute_benchmarks(df_train, df_test, episode_start, episode_length):
    episode_slice = slice(episode_start, episode_start + episode_length)
    spot_rets = df_test['spot_ret'].iloc[episode_slice]
    fut_rets = df_test['fut_ret'].iloc[episode_slice]

    # Unhedged
    var_unhedged = np.var(spot_rets)

    # Fixed ratio hedging (h=1)
    hedged_fixed = spot_rets - 1 * fut_rets
    var_fixed = np.var(hedged_fixed)
    effectiveness_fixed = 1 - (var_fixed / var_unhedged) if var_unhedged != 0 else 0

    # OLS/MVHR (static hedge ratio from train data)
    X_train = df_train[['fut_ret']].dropna()
    y_train = df_train['spot_ret'].loc[X_train.index]
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    h_ols = lr.coef_[0]
    
    hedged_ols = spot_rets - h_ols * fut_rets
    var_ols = np.var(hedged_ols)
    effectiveness_ols = 1 - (var_ols / var_unhedged) if var_unhedged != 0 else 0
    
    return {
        'unhedged': {'var': var_unhedged},
        'fixed': {'effectiveness': effectiveness_fixed, 'var': var_fixed, 'h': 1, 'returns': hedged_fixed},
        'ols_mvhr': {'effectiveness': effectiveness_ols, 'var': var_ols, 'h': h_ols, 'returns': hedged_ols}
    }

# Step 5: Visualization
def plot_results(hedged_rl, benchmarks, actions):
    fig, axs = plt.subplots(3, 1, figsize=(14, 20), sharex=True)
    
    axs[0].plot(hedged_rl, label=f'RL Hedged (Var: {np.var(hedged_rl):.6f})', color='blue')
    axs[0].plot(benchmarks['fixed']['returns'].values, label=f"Fixed Ratio (h=1) (Var: {benchmarks['fixed']['var']:.6f})", color='orange', linestyle='--')
    axs[0].plot(benchmarks['ols_mvhr']['returns'].values, label=f"OLS/MVHR (h={benchmarks['ols_mvhr']['h']:.2f}) (Var: {benchmarks['ols_mvhr']['var']:.6f})", color='green', linestyle='--')
    axs[0].set_title('Cumulative Hedged Returns Comparison')
    axs[0].set_ylabel('Cumulative Return')
    axs[0].legend()
    
    axs[1].plot(np.cumsum(hedged_rl), label='RL Hedged', color='blue')
    axs[1].plot(np.cumsum(benchmarks['fixed']['returns'].values), label='Fixed Ratio (h=1)', color='orange', linestyle='--')
    axs[1].plot(np.cumsum(benchmarks['ols_mvhr']['returns'].values), label=f"OLS/MVHR (h={benchmarks['ols_mvhr']['h']:.2f})", color='green', linestyle='--')
    axs[1].set_title('Cumulative Hedged Returns')
    axs[1].set_ylabel('Cumulative Return')
    axs[1].legend()

    axs[2].plot(actions, label='RL Hedge Ratio', color='purple', drawstyle='steps-post')
    axs[2].axhline(y=benchmarks['ols_mvhr']['h'], color='green', linestyle='--', label=f"OLS/MVHR Ratio ({benchmarks['ols_mvhr']['h']:.2f})")
    axs[2].axhline(y=1.0, color='orange', linestyle='--', label='Fixed Ratio (1.0)')
    axs[2].set_title('Dynamic Hedge Ratios from RL Agent')
    axs[2].set_xlabel('Time Step in Episode')
    axs[2].set_ylabel('Hedge Ratio')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig('logs/hedging_results.png')
    plt.show()

# NEW: Function to plot learning curve
def plot_learning_curve(log_dir='logs'):
    eval_path = os.path.join(log_dir, 'evaluations.npz')
    if not os.path.exists(eval_path):
        print("Evaluation log file not found. Skipping learning curve plot.")
        return

    data = np.load(eval_path)
    timesteps = data['timesteps']
    results = data['results']

    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, results)
    plt.title('Learning Curve')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'learning_curve.png'))
    plt.show()


# End-to-End Pipeline
if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Load pre-split data
    df_train, df_val, df_test = load_split_data(data_dir='data/rl_ready')
    
    # Train the agent
    model_path = 'models/ppo_hedging_model.zip' # CHANGED
    model = train_agent(df_train, df_val, model_path=model_path)
    
    # Plot learning curve
    plot_learning_curve() # NEW

    # Evaluate on the test set
    effectiveness_rl, var_rl, hedged_rl, actions, start_step = evaluate_agent(model, df_test)
    
    # Compute benchmarks for the same episode
    benchmarks = compute_benchmarks(df_train, df_test, start_step, len(hedged_rl))
    
    # Print results
    print("\n--- Hedging Performance Evaluation ---")
    print(f"Test Episode Start Step: {start_step}")
    print("-" * 35)
    print(f"Unhedged Portfolio Variance: {benchmarks['unhedged']['var']:.6f}")
    print("-" * 35)
    print("RL Agent (PPO):") # CHANGED
    print(f"  - Variance: {var_rl:.6f}")
    print(f"  - Effectiveness (VRE): {effectiveness_rl:.4f}")
    print("-" * 35)
    print("Benchmark - Fixed Ratio (h=1):")
    print(f"  - Variance: {benchmarks['fixed']['var']:.6f}")
    print(f"  - Effectiveness (VRE): {benchmarks['fixed']['effectiveness']:.4f}")
    print("-" * 35)
    print(f"Benchmark - OLS/MVHR (h={benchmarks['ols_mvhr']['h']:.4f}):")
    print(f"  - Variance: {benchmarks['ols_mvhr']['var']:.6f}")
    print(f"  - Effectiveness (VRE): {benchmarks['ols_mvhr']['effectiveness']:.4f}")
    print("-" * 35)
    
    # Plot results
    plot_results(hedged_rl, benchmarks, actions)
