import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pickle
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment.hedge_env import HedgeEnv

# --- Agent Creation Functions ---

def create_profit_maximization_agent(env, log_dir="./logs/profit_max/"):
    """
    Creates a PPO agent focused on maximizing profit.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Wrap env with Monitor to record episode rewards/lengths
    monitored_env = Monitor(env, log_dir)

    # Vectorize the monitored env
    vec_env = DummyVecEnv([lambda: monitored_env])
    
    # Normalize the environment
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # old code
    # model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir, use_sde=True)
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir, use_sde=True, ent_coef=0.01)
    return model

def create_risk_reduction_agent(env, log_dir="./logs/risk_reduction/"):
    """
    Creates a PPO agent focused on reducing risk.

    The reward function for this agent in the HedgeEnv should be based on a
    risk-adjusted return metric like the Sharpe Ratio or Sortino Ratio.
    The agent will be penalized for high volatility.

    Example Reward (using Sharpe Ratio):
    Reward = (step_pnl - risk_free_rate) / (std_dev_of_returns + 1e-6)
    """
    os.makedirs(log_dir, exist_ok=True)
    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir)
    return model

def create_cost_control_agent(env, log_dir="./logs/cost_control/"):
    """
    Creates a PPO agent focused on controlling transaction costs.

    The reward function for this agent in the HedgeEnv should be inversely
    proportional to the transaction costs incurred. The agent is penalized
    for frequent and large changes in futures positions.

    Example Reward:
    Reward = -transaction_cost
    """
    os.makedirs(log_dir, exist_ok=True)
    vec_env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_dir)
    return model

# --- Main Training Script ---

def train_and_evaluate():
    """
    Main function to load data, initialize the environment, create agents,
    train them, and save the models.
    """
    # 1. Load Data and Models
    try:
        data = pd.read_csv('data/fcpo_daily.csv', parse_dates=['datetime'])
        data = data.drop(['symbol'], axis=1)
        data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True).astype(int) / 10**9
        # data.rename(columns={'datetime': 'Date', 'close': 'Price'}, inplace=True)
    except FileNotFoundError:
        print("Error: data/fcpo_daily.csv not found. Please ensure the data file exists.")
        return

    try:
        with open('models/nextday_svr_optimized_model.pkl', 'rb') as f:
            forecast_nextday_model = pickle.load(f)
    except FileNotFoundError:
        print("Error: models/nextday_svr_optimized_model.pkl not found. Please ensure the model file exists.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    try:
        with open('models/nextmonth_svr_optimized_model.pkl', 'rb') as f:
            forecast_nextmonth_model = pickle.load(f)
    except FileNotFoundError:
        print("Error: models/nextmonth_svr_optimized_model.pkl not found. Please ensure the model file exists.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return


    # 2. Initialize Environment
    # The environment needs to be configured for each agent's reward structure.
    # For now, we will use the default environment (profit maximization).
    # In a full implementation, you would create different env instances
    # with different reward logic.
    env = HedgeEnv(
        data=data,
        forecast_nextday_model=forecast_nextday_model,
        forecast_nextmonth_model=forecast_nextmonth_model,
        start_date='2005-05-02',
        end_date='2021-08-16'
    )

    # 3. Create and Train Agents
    print("--- Training Profit Maximization Agent ---")
    profit_agent = create_profit_maximization_agent(env)
    profit_agent.learn(total_timesteps=100000)
    profit_agent.save("models/ppo_profit_maximization_agent")
    print("Profit Maximization Agent trained and saved.")

    # The following agents would require modifications to the environment's reward function.
    # This is a conceptual outline.

    # print("--- Training Risk Reduction Agent ---")
    # risk_env = HedgeEnv(...) # With risk-based reward
    # risk_agent = create_risk_reduction_agent(risk_env)
    # risk_agent.learn(total_timesteps=10000)
    # risk_agent.save("models/ppo_risk_reduction_agent")
    # print("Risk Reduction Agent trained and saved.")

    # print("--- Training Cost Control Agent ---")
    # cost_env = HedgeEnv(...) # With cost-based reward
    # cost_agent = create_cost_control_agent(cost_env)
    # cost_agent.learn(total_timesteps=10000)
    # cost_agent.save("models/ppo_cost_control_agent")
    # print("Cost Control Agent trained and saved.")

if __name__ == '__main__':
    # Note: You may need to install PyTorch, Gymnasium, and Stable-Baselines3
    # pip install torch gymnasium stable-baselines3 scikit-learn
    train_and_evaluate()