
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

def evaluate_agent():
    # 1. Load Data and Models
    try:
        data = pd.read_csv('data/fcpo_daily.csv', parse_dates=['datetime'])
        data = data.drop(['symbol'], axis=1)
        data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True).astype(int) / 10**9
    except FileNotFoundError:
        print("Error: data/fcpo_daily.csv not found.")
        return

    try:
        with open('models/nextday_svr_optimized_model.pkl', 'rb') as f:
            forecast_nextday_model = pickle.load(f)
        with open('models/nextmonth_svr_optimized_model.pkl', 'rb') as f:
            forecast_nextmonth_model = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading forecast model: {e}")
        return

    # 2. Load Trained Agent
    try:
        model = PPO.load("models/ppo_profit_maximization_agent.zip")
    except FileNotFoundError:
        print("Error: models/ppo_profit_maximization_agent.zip not found.")
        return

    # 3. Initialize Evaluation Environment
    # Use a different period for evaluation, e.g., 2024
    eval_env = HedgeEnv(
        data=data,
        forecast_nextday_model=forecast_nextday_model,
        forecast_nextmonth_model=forecast_nextmonth_model,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )

    # 4. Run Evaluation
    obs, info = eval_env.reset()
    done = False
    
    results = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        
        # Get action distribution to calculate confidence metrics
        # Note: the distribution is a DiagGaussianDistribution. Useful signals are:
        # - log_prob(action): higher means the action is more likely under the policy
        # - entropy(): lower entropy means the policy is more certain
        # - std (per-dim): small std means less spread in the Gaussian
        obs_tensor = torch.as_tensor(obs).to(model.policy.device)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor.reshape(1, -1))

            # prepare action tensor shaped (1, action_dim)
            action_tensor = torch.as_tensor(action).to(model.policy.device).reshape(1, -1)

            # log probability of the chosen action under the policy distribution
            log_prob_t = dist.log_prob(action_tensor)  # shape: (1,)
            # entropy of the distribution (lower -> more certain)
            entropy_t = dist.entropy()  # shape: (1,)

            # standard deviation per action-dimension (if available)
            # dist.std is provided by SB3's DiagGaussianDistribution
            try:
                std_arr = dist.std.detach().cpu().numpy().ravel()
            except Exception:
                std_arr = None

            # mean of the distribution (if available)
            try:
                mean_arr = dist.mean.detach().cpu().numpy().ravel()
            except Exception:
                mean_arr = None

            log_prob = float(log_prob_t.cpu().numpy().item())
            entropy = float(entropy_t.cpu().numpy().item())

            # Mahalanobis distance (squared) and normalized confidence score
            # For a diagonal Gaussian, mahal_sq = sum(((action - mean)/std)**2).
            # We map this to a [0,1] confidence via exp(-0.5 * mahal_sq), which
            # equals the Gaussian density at that point (up to a constant factor).
            try:
                if (mean_arr is not None) and (std_arr is not None):
                    act_np = action_tensor.cpu().numpy().ravel()
                    # avoid division by zero
                    denom = np.where(std_arr <= 0, 1e-8, std_arr)
                    mahal_sq = float(np.sum(((act_np - mean_arr) / denom) ** 2))
                    confidence = float(np.exp(-0.5 * mahal_sq))
                else:
                    mahal_sq = None
                    confidence = None
            except Exception:
                mahal_sq = None
                confidence = None

        hedge_ratio = float(action[0])

        obs, reward, done, truncated, info = eval_env.step(action)

        results.append({
            'timestep': info['current_step'],
            'portfolio_value': info['portfolio_value'],
            'hedge_ratio': hedge_ratio,
            'log_prob': log_prob,
            'entropy': entropy,
            'std0': float(std_arr[0]) if (std_arr is not None and len(std_arr) > 0) else None,
            'mahal_sq': mahal_sq,
            'confidence': confidence,
            'reward': reward
        })

    # 5. Report and Visualize Performance
    results_df = pd.DataFrame(results)
    
    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("--- Evaluation Complete ---")
    print(f"Final Portfolio Value: {results_df['portfolio_value'].iloc[-1]:,.2f}")
    
    # Save results to CSV
    results_df.to_csv(f"evaluation_results_{dt_str}.csv", index=False)
    print("Evaluation results saved to evaluation_results.csv")

    # Plot portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['timestep'], results_df['portfolio_value'])
    plt.title("Agent Portfolio Value Over Time (Evaluation)")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    filename = f"evaluation_performance_{dt_str}.png"
    plt.savefig(filename)
    print("Performance plot saved to evaluation_performance.png")
    plt.show()


if __name__ == '__main__':
    evaluate_agent()
