import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# ==============================================================================
# 1. DATA LOADING & PREPARATION
# ==============================================================================
def load_full_dataset(base_path='v3/data'):
    """
    Loads and combines the full dataset, mirroring the v3 notebook.
    This function is a placeholder; adapt it to load the feature-rich
    dataframe you created in your notebook.
    """
    # In the v3 notebook, you combined train, val, and test sets.
    # We do the same here to get the full, feature-rich dataset.
    # Let's assume 'palm_oil_data_cleaned.csv' is the final, processed file.
    # If not, you can recreate it by concatenating train.csv, val.csv, and test.csv
    # and re-running your feature engineering from the notebook.
    
    # Using 'palm_oil_data_cleaned.csv' as it appears to be a processed file in v3/data
    full_df_path = os.path.join(base_path, 'palm_oil_enriched_data.csv')
    if not os.path.exists(full_df_path):
        raise FileNotFoundError(
            "Please ensure 'palm_oil_enriched_data.csv' exists in 'v3/data/' "
            "or adapt this function to load your data."
        )
        
    df = pd.read_csv(full_df_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Drop any remaining NaNs just in case
    df.dropna(inplace=True)
    
    print(f"Successfully loaded data with shape: {df.shape}")
    print(f"Data contains {len(df.columns)} columns (features).")
    
    return df

# ==============================================================================
# 2. THE NEW, IMPROVED HEDGING ENVIRONMENT
# ==============================================================================
class HedgeEnvV3(gym.Env):
    """
    An environment optimized for COST CONTROL.
    The reward function is specifically designed to penalize transaction costs heavily.
    """
    def __init__(self, df, features, episode_length=90, transaction_cost=0.0005, reward_alpha=0.1, reward_beta=10.0):
        super(HedgeEnvV3, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.features = features
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.max_steps = len(df) - episode_length - 1

        # Action space: Continuous hedge ratio from 0 to 1.5
        self.action_space = spaces.Box(low=0.0, high=1.5, shape=(1,), dtype=np.float32)

        # Observation space: Market features + current hedge ratio
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features) + 1,), dtype=np.float32
        )
        
        self.current_hedge_ratio = 0.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        
        self.start_step = self.np_random.integers(0, self.max_steps)
        self.current_step = 0
        
        self.current_hedge_ratio = 0.0
        self.total_pnl = 0
        
        return self._get_obs(), {}

    def step(self, action):
        new_hedge_ratio = action[0]
        
        # === COST CONTROL REWARD FUNCTION ===
        abs_step = self.start_step + self.current_step
        
        # Calculate PnL for the step
        spot_return = self.df['spot_ret'].iloc[abs_step]
        futures_return = self.df['fut_ret'].iloc[abs_step]
        pnl = spot_return - self.current_hedge_ratio * futures_return
        
        # Calculate transaction cost for changing the hedge ratio
        cost = self.transaction_cost * abs(new_hedge_ratio - self.current_hedge_ratio)
        
        # The reward heavily penalizes cost, with a small incentive for performance
        reward = (self.reward_alpha * pnl) - (self.reward_beta * cost)
        
        # Update state for the next step
        self.current_hedge_ratio = new_hedge_ratio
        self.total_pnl += pnl # Track true PnL, not the skewed reward
        self.current_step += 1
        
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        obs = self._get_obs()
        info = {'pnl': pnl, 'cost': cost, 'total_pnl': self.total_pnl}
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        abs_step = self.start_step + self.current_step
        if abs_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        market_features = self.df[self.features].iloc[abs_step].values.astype(np.float32)
        obs = np.concatenate([market_features, [self.current_hedge_ratio]])
        return obs

# ==============================================================================
# 3. TRAINING & EVALUATION PIPELINE
# ==============================================================================
def main():
    # Create directories if they don't exist
    os.makedirs('v3/models', exist_ok=True)
    os.makedirs('v3/logs', exist_ok=True)

    # --- Data Loading ---
    full_df = load_full_dataset()
    
    # --- Feature Selection ---
    # Use the rich feature set from your v3 notebook, excluding identifiers and target
    # This is a sample list, adjust it to match your notebook's final feature set
    features = [col for col in full_df.columns if col not in [
        'date', 'spot_close', 'fut_close', 'next_day_pred', 'next_month_pred', 
        'hedge_ratio_target', 'target_hr', 'target_dhr', 'hr_now' # Exclude targets and identifiers
    ]]
    # Ensure all selected features are numeric
    features = [f for f in features if pd.api.types.is_numeric_dtype(full_df[f])]
    print(f"Using {len(features)} features for the agent's state.")
    
    # --- Generate unique filename for this training run ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # --- Save features for consistent evaluation ---
    features_path = f'v3/models/features_cost_control_{timestamp}.json'
    with open(features_path, 'w') as f:
        json.dump(features, f)
    print(f"Saved feature list to {features_path}")

    # --- Train/Test Split ---
    train_size = int(len(full_df) * 0.8)
    df_train = full_df.iloc[:train_size]
    df_test = full_df.iloc[train_size:]

    # --- Environment Setup ---
    # IMPORTANT: Wrap the environment with VecNormalize
    train_env = DummyVecEnv([lambda: HedgeEnvV3(df_train, features=features, reward_alpha=0.1, reward_beta=10.0)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env_raw = DummyVecEnv([lambda: HedgeEnvV3(df_test, features=features, reward_alpha=0.1, reward_beta=10.0)])
    eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False, clip_obs=10., training=False)


    # --- Agent Training ---
    model_path = f'v3/models/ppo_hedge_cost_control_{timestamp}.zip'
    vec_norm_path = f'v3/models/vec_normalize_cost_control_{timestamp}.pkl'
    
    eval_callback = EvalCallback(eval_env, best_model_save_path='v3/logs/best_model_cost_control',
                                 log_path='v3/logs/results/cost_control', eval_freq=10000,
                                 n_eval_episodes=20, deterministic=True)

    # PPO is a great choice for this continuous control problem
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,  # Increased for more exploration
        tensorboard_log="v3/logs/tensorboard_cost_control/"
    )
    
    print("\n--- Starting Agent Training ---")
    # Increased total_timesteps for better learning
    model.learn(total_timesteps=500000, callback=eval_callback)
    
    # --- IMPORTANT: Save the model and the normalization stats ---
    model.save(model_path)
    train_env.save(vec_norm_path)
    print(f"--- Training Complete. Model saved to {model_path} and normalization stats to {vec_norm_path} ---")

    # --- Evaluation ---
    print("\n--- Evaluating Cost Control Agent on Test Set ---")
    
    # Load the normalization stats and apply them to the evaluation env
    eval_env = VecNormalize.load(vec_norm_path, eval_env_raw)
    eval_env.training = False # Important: do not update moving averages
    eval_env.norm_reward = False # Do not normalize rewards for evaluation

    # Load the trained model
    model = PPO.load(model_path, env=eval_env)

    obs = eval_env.reset()
    
    # --- Store detailed results for plotting ---
    episode_rewards = []
    episode_pnl = []
    episode_costs = []
    hedge_ratios = []
    
    # Use the length of the raw dataframe for the loop
    for _ in range(len(df_test) - 91): 
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        
        episode_rewards.append(reward[0])
        episode_pnl.append(info[0]['pnl'])
        episode_costs.append(info[0]['cost'])
        hedge_ratios.append(eval_env.envs[0].current_hedge_ratio)
        
        if done:
            break
            
    # --- Plotting Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(3, 1, figsize=(15, 18), sharex=True)

    # Plot 1: Cumulative PnL (The actual profit/loss)
    axs[0].plot(np.cumsum(episode_pnl), label='Agent Cumulative PnL', color='green')
    axs[0].set_title('Cost Control Agent: Actual PnL on Test Data', fontsize=16)
    axs[0].set_ylabel('Cumulative PnL', fontsize=12)
    axs[0].legend()

    # Plot 2: Cumulative Transaction Costs
    axs[1].plot(np.cumsum(episode_costs), label='Agent Cumulative Costs', color='red')
    axs[1].set_title('Cost Control Agent: Cumulative Transaction Costs', fontsize=16)
    axs[1].set_ylabel('Cumulative Costs', fontsize=12)
    axs[1].legend()

    # Plot 3: Hedge Ratios Chosen by the Agent
    axs[2].plot(hedge_ratios, label='RL Agent Hedge Ratio', color='purple', drawstyle='steps-post')
    axs[2].set_title('Cost Control Agent: Dynamic Hedge Ratio Policy', fontsize=16)
    axs[2].set_ylabel('Hedge Ratio', fontsize=12)
    axs[2].set_xlabel('Time Step', fontsize=12)
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'v3/logs/v3_cost_control_evaluation_results_{timestamp}.png')
    print(f"\n--- Evaluation complete. Results plot saved to v3/logs/v3_cost_control_evaluation_results_{timestamp}.png ---")
    plt.show()


if __name__ == "__main__":
    main()
