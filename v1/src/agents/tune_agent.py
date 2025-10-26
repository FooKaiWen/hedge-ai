import sys
import os
import argparse
import pickle
import pandas as pd
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.environment.hedge_env import HedgeEnv

# This map is now consistent with evaluate_agent.py
reward_strategy_map = {'profit': 'profit', 'risk': 'sharpe', 'cost': 'cost'}

def objective(trial, agent_type):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    # 1. Suggest Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    risk_aversion = trial.suggest_float("risk_aversion", 0.0, 0.1)

    # 2. Load Data and Models
    try:
        data = pd.read_csv('data/fcpo_daily.csv', parse_dates=['datetime'])
        data = data.drop(['symbol'], axis=1)
        data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True).astype(int) / 10**9
    except FileNotFoundError:
        print("Error: data/fcpo_daily.csv not found.")
        # Pruning the trial since data is not available
        raise optuna.exceptions.TrialPruned()

    try:
        with open('models/nextday_svr_optimized_model.pkl', 'rb') as f:
            forecast_nextday_model = pickle.load(f)
        with open('models/nextmonth_svr_optimized_model.pkl', 'rb') as f:
            forecast_nextmonth_model = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading forecast model: {e}")
        # Pruning the trial since models are not available
        raise optuna.exceptions.TrialPruned()

    # 3. Create and Train the Agent
    reward_strategy = reward_strategy_map[agent_type]
    
    train_env = HedgeEnv(
        data=data,
        forecast_nextday_model=forecast_nextday_model,
        forecast_nextmonth_model=forecast_nextmonth_model,
        start_date='2005-05-02',
        end_date='2021-08-16',
        reward_strategy=reward_strategy,
        risk_aversion=risk_aversion
    )
    monitored_env = Monitor(train_env)
    train_vec_env = DummyVecEnv([lambda: monitored_env])
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        train_vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=0
    )
    try:
        model.learn(total_timesteps=50000) # Using a shorter training time for tuning
    except ValueError:
        # For example, if n_steps is too large for the environment
        raise optuna.exceptions.TrialPruned()


    # 4. Evaluate the Agent
    # Create a temporary path for normalization stats to avoid race conditions
    stats_path = f"temp_tuning_stats_{trial.number}.pkl"
    train_vec_env.save(stats_path)

    eval_env = HedgeEnv(
        data=data,
        forecast_nextday_model=forecast_nextday_model,
        forecast_nextmonth_model=forecast_nextmonth_model,
        start_date='2021-08-17',
        end_date='2025-09-11',
        reward_strategy=reward_strategy,
        risk_aversion=risk_aversion
    )

    eval_vec_env = DummyVecEnv([lambda: Monitor(eval_env)])
    eval_vec_env = VecNormalize.load(stats_path, eval_vec_env)
    eval_vec_env.training = False
    eval_vec_env.norm_reward = False

    obs = eval_vec_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_vec_env.step(action)
        done = dones[0]
    
    final_portfolio_value = infos[0]['portfolio_value']

    # Clean up the temporary file
    os.remove(stats_path)

    return final_portfolio_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tune hyperparameters for a specialized PPO agent.")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=['profit', 'risk', 'cost'],
        help="The type of agent to tune."
    )
    args = parser.parse_args()
    agent_type = args.agent

    study = optuna.create_study(direction="maximize")
    # Use a lambda to pass the agent_type to the objective function
    try:
        study.optimize(lambda trial: objective(trial, agent_type), n_trials=100)
    except KeyboardInterrupt:
        print("Tuning interrupted by user.")


    print(f"\n--- Hyperparameter Tuning Complete for {agent_type.title()} Agent ---")
    
    if study.best_trial:
        print(f"Best trial number: {study.best_trial.number}")
        print(f"Best value (final portfolio value): {study.best_value:,.2f}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

        # Save the best hyperparameters to an agent-specific file
        hyperparams_path = f"models/{agent_type}_agent_best_hyperparameters.pkl"
        with open(hyperparams_path, "wb") as f:
            pickle.dump(study.best_params, f)
        print(f"\nBest hyperparameters saved to {hyperparams_path}")
    else:
        print("No successful trials were completed.")