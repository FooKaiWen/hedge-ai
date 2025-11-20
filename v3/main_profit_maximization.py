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
    This new environment fixes the critical flaws from the v2 attempts.
    """
    def __init__(self, df, features, episode_length=90, transaction_cost=0.0005):
        super(HedgeEnvV3, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.features = features
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.max_steps = len(df) - episode_length - 1

        # === KEY IMPROVEMENT 1: CONTINUOUS ACTION SPACE ===
        # The agent can choose any precise hedge ratio between 0 and 1.5
        self.action_space = spaces.Box(low=0.0, high=1.5, shape=(1,), dtype=np.float32)

        # === KEY IMPROVEMENT 2: RICH & COMPLETE STATE SPACE ===
        # The observation includes all market features PLUS the agent's current position.
        # Shape is num_features + 1 (for current_hedge_ratio)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features) + 1,), dtype=np.float32
        )
        
        self.current_hedge_ratio = 0.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        
        # Start at a random point in the data for robust training
        self.start_step = self.np_random.integers(0, self.max_steps)
        self.current_step = 0
        
        # Reset portfolio state
        self.current_hedge_ratio = 0.0
        self.total_pnl = 0
        
        return self._get_obs(), {}

    def step(self, action):
        new_hedge_ratio = action[0]
        
        # === KEY IMPROVEMENT 3: REALISTIC REWARD FUNCTION ===
        # Reward is based on PnL minus transaction costs.
        
        abs_step = self.start_step + self.current_step
        
        # Calculate PnL for the step
        spot_return = self.df['spot_ret'].iloc[abs_step]
        futures_return = self.df['fut_ret'].iloc[abs_step]
        
        # PnL is calculated based on the position held *during* the step (the previous ratio)
        pnl = spot_return - self.current_hedge_ratio * futures_return
        
        # Calculate transaction cost for changing the hedge ratio
        cost = self.transaction_cost * abs(new_hedge_ratio - self.current_hedge_ratio)
        
        # The reward is the net PnL after costs
        reward = pnl - cost
        
        # Update state for the next step
        self.current_hedge_ratio = new_hedge_ratio
        self.total_pnl += reward
        self.current_step += 1
        
        # Check if the episode is done
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        obs = self._get_obs()
        info = {'pnl': pnl, 'cost': cost, 'total_pnl': self.total_pnl}
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # === KEY IMPROVEMENT 4: STATE INCLUDES AGENT'S POSITION ===
        # The agent MUST know its current hedge ratio to make an informed decision.
        
        abs_step = self.start_step + self.current_step
        if abs_step >= len(self.df):
            # Return a zero observation if at the end of the dataframe
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        market_features = self.df[self.features].iloc[abs_step].values.astype(np.float32)
        
        # Concatenate market features with the agent's current position
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
    features_path = f'v3/models/features_profit_maximization_{timestamp}.json'
    with open(features_path, 'w') as f:
        json.dump(features, f)
    print(f"Saved feature list to {features_path}")

    # --- Train/Test Split ---
    train_size = int(len(full_df) * 0.8)
    df_train = full_df.iloc[:train_size]
    df_test = full_df.iloc[train_size:]

    # --- Environment Setup ---
    # IMPORTANT: Wrap the environment with VecNormalize
    train_env = DummyVecEnv([lambda: HedgeEnvV3(df_train, features=features)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env_raw = DummyVecEnv([lambda: HedgeEnvV3(df_test, features=features)])
    eval_env = VecNormalize(eval_env_raw, norm_obs=True, norm_reward=False, clip_obs=10., training=False)


    # --- Agent Training ---
    model_path = f'v3/models/ppo_hedge_profit_maximization_{timestamp}.zip'
    vec_norm_path = f'v3/models/vec_normalize_profit_maximization_{timestamp}.pkl'
    
    eval_callback = EvalCallback(eval_env, best_model_save_path='v3/logs/best_model_profit_maximization',
                                 log_path='v3/logs/results/profit_maximization', eval_freq=10000,
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
        tensorboard_log="v3/logs/tensorboard_profit_maximization/"
    )
    
    print("\n--- Starting Agent Training ---")
    # Increased total_timesteps for better learning
    model.learn(total_timesteps=500000, callback=eval_callback)
    
    # --- IMPORTANT: Save the model and the normalization stats ---
    model.save(model_path)
    train_env.save(vec_norm_path)
    print(f"--- Training Complete. Model saved to {model_path} and normalization stats to {vec_norm_path} ---")

    # --- Evaluation ---
    print("\n--- Evaluating Trained Agent on Test Set ---")
    
    # Load the normalization stats and apply them to the evaluation env
    eval_env = VecNormalize.load(vec_norm_path, eval_env_raw)
    eval_env.training = False # Important: do not update moving averages
    eval_env.norm_reward = False # Do not normalize rewards for evaluation

    # Load the trained model
    model = PPO.load(model_path, env=eval_env)

    obs = eval_env.reset()
    episode_rewards = []
    hedge_ratios = []
    
    # Use the length of the raw dataframe for the loop
    for _ in range(len(df_test) - 91): 
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = eval_env.step(action)
        episode_rewards.append(reward[0])
        # Access the current hedge ratio from the environment's internal state
        hedge_ratios.append(eval_env.envs[0].current_hedge_ratio)
        if done:
            break
            
    # --- Plotting Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    # Plot 1: Cumulative Rewards (Total PnL)
    axs[0].plot(np.cumsum(episode_rewards), label='RL Agent Cumulative PnL', color='blue')
    axs[0].set_title('Agent Performance on Test Data', fontsize=16)
    axs[0].set_ylabel('Cumulative Reward (PnL)', fontsize=12)
    axs[0].legend()

    # Plot 2: Hedge Ratios Chosen by the Agent
    axs[1].plot(hedge_ratios, label='RL Agent Hedge Ratio', color='purple', drawstyle='steps-post')
    axs[1].set_title('Dynamic Hedge Ratio Policy', fontsize=16)
    axs[1].set_ylabel('Hedge Ratio', fontsize=12)
    axs[1].set_xlabel('Time Step', fontsize=12)
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig('v3/logs/v3_profit_maximization_evaluation_results.png')
    print("\n--- Evaluation complete. Results plot saved to v3/logs/v3_profit_maximization_evaluation_results.png ---")
    plt.show()


if __name__ == "__main__":
    main()
