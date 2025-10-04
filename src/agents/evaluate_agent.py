import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import pickle
from stable_baselines3 import PPO
from src.environment.hedge_env import HedgeEnv
import matplotlib.pyplot as plt
import torch
import numpy as np
from datetime import datetime

def run_evaluation(model, eval_env, save_results=True):
    """
    Evaluates a trained agent in a given environment.

    Args:
        model: The trained agent model.
        eval_env: The evaluation environment.
        save_results (bool): Whether to save the evaluation results to a file.

    Returns:
        float: The final portfolio value.
    """
    obs, info = eval_env.reset()
    done = False
    
    results = []

    while not done:
        action, _states = model.predict(obs, deterministic=False)
        
        obs, reward, done, truncated, info = eval_env.step(action)

        results.append({
            'timestep': info['current_step'],
            'portfolio_value': info['portfolio_value'],
            'hedge_ratio': float(action[0]),
            'reward': reward
        })

    results_df = pd.DataFrame(results)
    final_portfolio_value = results_df['portfolio_value'].iloc[-1]

    if save_results:
        dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("--- Evaluation Complete ---")
        print(f"Final Portfolio Value: {final_portfolio_value:,.2f}")
        
        # Save results to CSV
        results_df.to_csv(f"outputs/evaluation_results_{dt_str}.csv", index=False)
        print(f"Evaluation results saved to outputs/evaluation_results_{dt_str}.csv")

        # Plot portfolio value over time
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['timestep'], results_df['portfolio_value'])
        plt.title("Agent Portfolio Value Over Time (Evaluation)")
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        filename = f"outputs/evaluation_performance_{dt_str}.png"
        plt.savefig(filename)
        print(f"Performance plot saved to outputs/evaluation_performance_{dt_str}.png")
        plt.show()

    return final_portfolio_value

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

if __name__ == '__main__':
    # This block is for standalone execution of the evaluation script.
    # It now first trains a model with the best hyperparameters and then evaluates it.

    # 1. Load Data and Forecast Models
    try:
        data = pd.read_csv('data/fcpo_daily.csv', parse_dates=['datetime'])
        data = data.drop(['symbol'], axis=1)
        data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True).astype(int) / 10**9
    except FileNotFoundError:
        print("Error: data/fcpo_daily.csv not found.")
        sys.exit(1)

    try:
        with open('models/nextday_svr_optimized_model.pkl', 'rb') as f:
            forecast_nextday_model = pickle.load(f)
        with open('models/nextmonth_svr_optimized_model.pkl', 'rb') as f:
            forecast_nextmonth_model = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading forecast model: {e}")
        sys.exit(1)

    # 2. Load and Separate Hyperparameters
    try:
        with open("models/profit_maximization_best_hyperparameters.pkl.pkl", "rb") as f:
            best_hyperparameters = pickle.load(f)
    except FileNotFoundError:
        print("Error: models/profit_maximization_best_hyperparameters.pkl.pkl not found. Please run tune_agent.py first.")
        sys.exit(1)

    # Separate environment params from PPO params
    risk_aversion = best_hyperparameters.pop('risk_aversion', 0.0) # Pop with default
    ppo_hyperparameters = best_hyperparameters # The rest are for PPO
    
    print("--- Loaded Best Hyperparameters ---")
    print(f"    risk_aversion (for env): {risk_aversion}")
    print("    PPO Hyperparameters:")
    for key, value in ppo_hyperparameters.items():
        print(f"        {key}: {value}")

    # 3. Create and Train the Agent with Best Hyperparameters
    print("\n--- Training Agent with Best Hyperparameters ---")
    train_env = HedgeEnv(
        data=data,
        forecast_nextday_model=forecast_nextday_model,
        forecast_nextmonth_model=forecast_nextmonth_model,
        start_date='2005-05-02',
        end_date='2021-08-16',
        risk_aversion=risk_aversion
    )
    monitored_env = Monitor(train_env)
    vec_env = DummyVecEnv([lambda: monitored_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        **ppo_hyperparameters
    )
    model.learn(total_timesteps=500) # Increased timesteps for final training
    print("--- Training Complete ---")

    # Save the trained model
    model.save("models/ppo_agent_with_best_hyperparams.zip")
    print("Trained model saved to models/ppo_agent_with_best_hyperparams.zip")


    # 4. Initialize Evaluation Environment
    eval_env = HedgeEnv(
        data=data,
        forecast_nextday_model=forecast_nextday_model,
        forecast_nextmonth_model=forecast_nextmonth_model,
        start_date='2021-08-17',
        end_date='2025-09-11',
        risk_aversion=risk_aversion # Use the same risk aversion for eval
    )

    # 5. Run Evaluation
    print("\n--- Evaluating Trained Agent ---")
    run_evaluation(model, eval_env)