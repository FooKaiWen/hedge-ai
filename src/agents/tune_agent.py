
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pickle
import pandas as pd
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment.hedge_env import HedgeEnv
from src.agents.evaluate_agent import run_evaluation

def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    # 1. Suggest Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
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
        return -1e9

    try:
        with open('models/nextday_svr_optimized_model.pkl', 'rb') as f:
            forecast_nextday_model = pickle.load(f)
        with open('models/nextmonth_svr_optimized_model.pkl', 'rb') as f:
            forecast_nextmonth_model = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading forecast model: {e}")
        return -1e9

    # 3. Create and Train the Agent
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
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=0
    )
    model.learn(total_timesteps=50000) # Using a shorter training time for tuning

    # 4. Create Evaluation Environment
    eval_env = HedgeEnv(
        data=data,
        forecast_nextday_model=forecast_nextday_model,
        forecast_nextmonth_model=forecast_nextmonth_model,
        start_date='2021-08-17',
        end_date='2025-09-11',
        risk_aversion=risk_aversion
    )

    # 5. Evaluate the Agent
    final_portfolio_value = run_evaluation(model, eval_env, save_results=False)

    return final_portfolio_value

if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("--- Hyperparameter Tuning Complete ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best value (final portfolio value): {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Save the best hyperparameters
    with open("models/best_hyperparameters.pkl", "wb") as f:
        pickle.dump(study.best_params, f)
    print("Best hyperparameters saved to models/best_hyperparameters.pkl")
