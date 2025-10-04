import sys
import os
import argparse
import pandas as pd
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.environment.hedge_env import HedgeEnv

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a specialized PPO agent with its best hyperparameters.")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=['profit', 'risk', 'cost'],
        help="The type of agent to train and evaluate."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Number of timesteps to train the agent before evaluation."
    )
    args = parser.parse_args()
    agent_type = args.agent

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
    hyperparams_path = f"models/{agent_type}_agent_best_hyperparameters.pkl"
    try:
        with open(hyperparams_path, "rb") as f:
            best_hyperparameters = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Hyperparameter file not found at {hyperparams_path}.")
        print(f"Please run 'python src/agents/tune_agent.py --agent {agent_type}' first.")
        sys.exit(1)

    risk_aversion = best_hyperparameters.pop('risk_aversion', 0.0)
    ppo_hyperparameters = best_hyperparameters
    ppo_hyperparameters.setdefault('max_grad_norm', 0.5) # Add default grad clipping

    print(f"--- Loaded Best Hyperparameters for {agent_type.title()} Agent ---")
    for key, value in ppo_hyperparameters.items():
        print(f"    {key}: {value}")

    # 3. Create and Train the Agent
    print(f"\n--- Training {agent_type.title()} Agent with Best Hyperparameters ---")
    reward_strategy_map = {'profit': 'profit', 'risk': 'sharpe', 'cost': 'cost'}
    reward_strategy = reward_strategy_map[agent_type]
    tensorboard_log_path = f"./tensorboard_logs/{agent_type}_agent/"

    train_env = HedgeEnv(
        data=data,
        forecast_nextday_model=forecast_nextday_model,
        forecast_nextmonth_model=forecast_nextmonth_model,
        start_date='2005-05-02',
        end_date='2021-08-16',
        # reward_strategy=reward_strategy,
        risk_aversion=risk_aversion
    )
    
    # The Monitor wrapper is essential for SB3 logging
    monitored_env = Monitor(train_env, tensorboard_log_path)
    train_vec_env = DummyVecEnv([lambda: monitored_env])
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        train_vec_env,
        verbose=0,
        tensorboard_log=tensorboard_log_path, # Enable Tensorboard logging
        **ppo_hyperparameters
    )
    
    try:
        model.learn(total_timesteps=args.timesteps)
        print("--- Training Complete ---")
    except ValueError as e:
        print(f"\nERROR: Training failed with a ValueError: {e}")
        sys.exit(1)

    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the trained model and normalization stats
    model_save_path = f"models/evaluated_{agent_type}_agent_{dt_str}.zip"
    stats_path = os.path.join(tensorboard_log_path, "vec_normalize.pkl")
    model.save(model_save_path)
    train_vec_env.save(stats_path)
    print(f"Trained model saved to {model_save_path}")
    print(f"Normalization stats saved to {stats_path}")

    # 4. Initialize Evaluation Environment
    eval_env = HedgeEnv(
        data=data,
        forecast_nextday_model=forecast_nextday_model,
        forecast_nextmonth_model=forecast_nextmonth_model,
        start_date='2021-08-17',
        end_date='2025-09-11',
        # reward_strategy=reward_strategy,
        risk_aversion=risk_aversion
    )

    # Correctly apply normalization from training to evaluation
    eval_vec_env = DummyVecEnv([lambda: Monitor(eval_env)])
    eval_vec_env = VecNormalize.load(stats_path, eval_vec_env)
    eval_vec_env.training = False
    eval_vec_env.norm_reward = False

    # 5. Run Evaluation
    print(f"\n--- Evaluating Trained {agent_type.title()} Agent ---")
    obs = eval_vec_env.reset()
    done = False
    results = []
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_vec_env.step(action)
        done = dones[0]
        info = infos[0]
        results.append({
            'timestep': info['current_step'],
            'portfolio_value': info['portfolio_value'],
            'hedge_ratio': float(action[0]),
            'reward': rewards[0]
        })

    results_df = pd.DataFrame(results)
    final_portfolio_value = results_df['portfolio_value'].iloc[-1]
    
    print("--- Evaluation Complete ---")
    print(f"Final Portfolio Value: {final_portfolio_value:,.2f}")

    results_filename = f"outputs/{agent_type}_evaluation_results_{dt_str}.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"Evaluation results saved to {results_filename}")

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['timestep'], results_df['portfolio_value'])
    plt.title(f"{agent_type.title()} Agent Portfolio Value Over Time (Evaluation)")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plot_filename = f"outputs/{agent_type}_evaluation_performance_{dt_str}.png"
    plt.savefig(plot_filename)
    print(f"Performance plot saved to {plot_filename}")
    plt.show()

if __name__ == '__main__':
    main()