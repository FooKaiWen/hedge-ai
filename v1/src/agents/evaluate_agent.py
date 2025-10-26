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


def evaluate_model(model, data, forecast_nextday_model, forecast_nextmonth_model, agent_type):
    """
    Evaluate a trained model on the environment without any wrappers.
    """
    print(f"\n--- Evaluating {agent_type.title()} Agent ---")

    eval_env = HedgeEnv(
        data=data,
        forecast_nextday_model=forecast_nextday_model,
        forecast_nextmonth_model=forecast_nextmonth_model,
        start_date='2021-08-17',
        end_date='2025-09-11'
    )

    obs, _ = eval_env.reset()
    results = []

    for _ in range(eval_env.max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        results.append({
            'timestep': info['current_step'],
            'portfolio_value': info['portfolio_value'],
            'hedge_ratio': action.item(),
            'reward': reward
        })
        if terminated or truncated:
            break

    results_df = pd.DataFrame(results)
    final_value = results_df['portfolio_value'].iloc[-1]
    metrics = eval_env.compute_metrics()

    print(f"--- Evaluation Complete for {agent_type.title()} ---")
    print(f"Final Portfolio Value: {final_value:,.2f}")
    print(f"Metrics: ROI={metrics['ROI']:.6f}, Sharpe={metrics['Sharpe Ratio']:.4f}, CostEff={metrics['Cost Efficiency Ratio']:.4f}")

    return results_df, metrics

def train_and_evaluate(agent_type, data, forecast_nextday_model, forecast_nextmonth_model, timesteps):
    """
    Train and evaluate one agent type (profit, risk, or cost).
    Returns a tuple of (results_df, metrics_dict).
    """
    # Load best hyperparameters
    hyperparams_path = f"models/{agent_type}_agent_best_hyperparameters.pkl"
    try:
        with open(hyperparams_path, "rb") as f:
            best_hyperparameters = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Hyperparameter file not found at {hyperparams_path}.")
        sys.exit(1)

    risk_aversion = best_hyperparameters.pop('risk_aversion', 0.0)  # Kept for compatibility with old files
    ppo_hyperparameters = best_hyperparameters
    ppo_hyperparameters.setdefault('max_grad_norm', 0.5)

    print(f"\n--- Loaded Best Hyperparameters for {agent_type.title()} Agent ---")
    for k, v in ppo_hyperparameters.items():
        print(f"    {k}: {v}")
    print("INFO: All agents now train on the unified rolling Sharpe ratio reward signal.")

    tensorboard_log_path = f"./tensorboard_logs/{agent_type}_agent/"

    # --- Training Environment ---
    train_env = HedgeEnv(
        data=data,
        forecast_nextday_model=forecast_nextday_model,
        forecast_nextmonth_model=forecast_nextmonth_model,
        start_date='2005-05-02',
        end_date='2021-08-16'
    )

    monitored_env = Monitor(train_env, tensorboard_log_path)
    train_vec_env = DummyVecEnv([lambda: monitored_env])
    train_vec_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True)

    model = PPO(
        "MlpPolicy",
        train_vec_env,
        verbose=0,
        tensorboard_log=tensorboard_log_path,
        **ppo_hyperparameters
    )

    print(f"\n--- Training {agent_type.title()} Agent ---")
    model.learn(total_timesteps=timesteps)
    print("--- Training Complete ---")

    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = f"models/evaluated_{agent_type}_agent_{dt_str}.zip"
    stats_path = os.path.join(tensorboard_log_path, "vec_normalize.pkl")
    model.save(model_save_path)
    train_vec_env.save(stats_path)

    # --- Evaluation ---
    results_df, metrics = evaluate_model(model, data, forecast_nextday_model, forecast_nextmonth_model, agent_type)

    # Save outputs
    os.makedirs("outputs", exist_ok=True)
    results_filename = f"outputs/{agent_type}_evaluation_results_{dt_str}.csv"
    results_df.to_csv(results_filename, index=False)

    # Plot performance
    plt.figure(figsize=(10, 5))
    plt.plot(results_df['timestep'], results_df['portfolio_value'])
    plt.title(f"{agent_type.title()} Agent Portfolio Value")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plot_filename = f"outputs/{agent_type}_evaluation_performance_{dt_str}.png"
    plt.savefig(plot_filename)
    plt.close()

    return results_df, metrics



    


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate PPO agents (profit, risk, cost).")
    parser.add_argument(
        "--agent",
        type=str,
        default="all",
        choices=['profit', 'risk', 'cost', 'all'],
        help="Choose one agent or 'all' to evaluate all agents."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Number of timesteps for training each agent."
    )
    args = parser.parse_args()

    # --- Load data and models ---
    try:
        data = pd.read_csv('data/fcpo_daily.csv', parse_dates=['datetime'])
        data = data.drop(['symbol'], axis=1)
        data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=False).astype(int) / 10**9
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

    agent_list = ['profit', 'risk', 'cost'] if args.agent == 'all' else [args.agent]

    summary = []
    for agent_type in agent_list:
        results_df, metrics = train_and_evaluate(
            agent_type, data, forecast_nextday_model, forecast_nextmonth_model, args.timesteps
        )
        metrics['Agent'] = agent_type.title()
        summary.append(metrics)

    # --- Summarize all metrics ---
    summary_df = pd.DataFrame(summary)[['Agent', 'ROI', 'Sharpe Ratio', 'Cost Efficiency Ratio']]
    summary_path = "outputs/agent_performance_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n===== PERFORMANCE SUMMARY =====")
    print(summary_df)

    # --- Combined comparison plot ---
    plt.figure(figsize=(10, 6))
    for agent_type in agent_list:
        filename = max([f for f in os.listdir("outputs") if f.startswith(agent_type) and f.endswith(".csv")],
                       key=lambda x: os.path.getctime(os.path.join("outputs", x)))
        df = pd.read_csv(os.path.join("outputs", filename))
        plt.plot(df['timestep'], df['portfolio_value'], label=agent_type.title())

    plt.title("Agent Comparison: Portfolio Value Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/all_agents_comparison.png")
    plt.show()

    print(f"\nSummary saved to {summary_path}")
    print("Combined performance plot saved to outputs/all_agents_comparison.png")


if __name__ == '__main__':
    main()
