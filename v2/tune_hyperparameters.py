import optuna
import numpy as np
import os
import logging
import sys
from main import load_split_data, HedgingEnv, evaluate_agent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Set up logging to suppress informational messages from Optuna and Stable Baselines
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.get_logger("optuna").setLevel(logging.WARNING)


def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    # Load data
    # Using validation set for evaluation as is standard practice.
    df_train, df_val, _ = load_split_data(data_dir='data/rl_ready')

    # Define hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [256, 512, 1024, 2048])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 1.0)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_epochs = trial.suggest_int('n_epochs', 5, 20)


    # Create environment
    env_fn = lambda: HedgingEnv(df_train)
    env = DummyVecEnv([env_fn])

    # Create PPO model with suggested hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )

    # Train the model for a shorter duration for faster tuning
    model.learn(total_timesteps=30000)

    # Evaluate the model on the validation set
    # The metric to optimize is Variance Reduction Effectiveness (VRE)
    effectiveness_rl, _, _, _, _ = evaluate_agent(model, df_val)

    return effectiveness_rl

if __name__ == "__main__":
    # Create a study object and specify the direction is to maximize VRE
    study = optuna.create_study(direction='maximize')
    
    # Run the optimization
    # The plan suggests 50-100 trials. Let's go with 50.
    study.optimize(objective, n_trials=50, n_jobs=-1) # Use all available CPU cores

    print("\n--- Hyperparameter Optimization Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")
    
    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (VRE): {best_trial.value:.4f}")
    
    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # The next step according to the plan is to hard-code these into main.py
    print("\nNext step: Update the PPO hyperparameters in main.py with these values and retrain the model.")
