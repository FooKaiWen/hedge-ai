import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================
def load_full_dataset(base_path='v3/data'):
    """Loads and preprocesses the dataset."""
    full_df_path = os.path.join(base_path, 'palm_oil_enriched_data.csv')
    if not os.path.exists(full_df_path):
        raise FileNotFoundError(f"Ensure '{full_df_path}' exists.")
        
    df = pd.read_csv(full_df_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    return df

# ==============================================================================
# 2. METRICS CALCULATION
# ==============================================================================
def calculate_metrics(results):
    """Calculates summary statistics from simulation results."""
    pnl = np.array(results['pnls'])
    costs = np.array(results['costs'])
    hedge_ratios = np.array(results['hedge_ratios'])
    
    total_pnl = pnl.sum()
    total_cost = costs.sum()
    net_pnl = total_pnl - total_cost
    
    # Sharpe Ratio (annualized, assuming daily returns)
    # Use net pnl for sharpe ratio calculation
    net_pnl_steps = pnl - costs
    sharpe_ratio = (net_pnl_steps.mean() / (net_pnl_steps.std() + 1e-9)) * np.sqrt(252)
    
    # Portfolio Turnover
    turnover = np.sum(np.abs(np.diff(np.array([0.0] + hedge_ratios))))
    
    return {
        'Total PnL': total_pnl,
        'Net PnL': net_pnl,
        'Total Costs': total_cost,
        'Sharpe Ratio': sharpe_ratio,
        'Turnover': turnover
    }

# ==============================================================================
# 3. MAIN SIMULATION SCRIPT
# ==============================================================================
def main():
    print("--- Starting On-the-Spot Simulation ---")
    
    # --- 1. Load Data and Define Models ---
    full_df = load_full_dataset()
    _ , df_test = np.split(full_df, [int(len(full_df) * 0.8)])
    df_test = df_test.reset_index(drop=True)

    agents_to_load = {
        "Cost Control": {
            "model_path": "v3/models/ppo_hedge_cost_control_20251126_221942.zip",
            "vec_norm_path": "v3/models/vec_normalize_cost_control_20251126_221942.pkl",
            "features_path": "v3/models/features_cost_control_20251126_221942.json"
        },
        "Profit Maximization": {
            "model_path": "v3/models/ppo_hedge_profit_maximization_20251118_223523.zip",
            "vec_norm_path": "v3/models/vec_normalize_profit_maximization_20251118_223523.pkl",
            "features_path": "v3/models/features_profit_maximization_20251118_223523.json"
        },
        "Risk Reduction": {
            "model_path": "v3/models/ppo_hedge_risk_reduction_20251118_223523.zip",
            "vec_norm_path": "v3/models/vec_normalize_risk_reduction_20251118_223523.pkl",
            "features_path": "v3/models/features_risk_reduction_20251118_223523.json"
        }
    }
    
    # Define a mock environment to satisfy VecNormalize.load() requirements
    class MockEnv(gym.Env):
        def __init__(self, features):
            super().__init__()
            self.action_space = spaces.Box(low=0.0, high=1.5, shape=(1,), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(features) + 1,), dtype=np.float32
            )
        def step(self, action): pass
        def reset(self, *, seed=None, options=None): pass

    # Load models and their associated normalization stats
    agent_models = {}
    for name, paths in agents_to_load.items():
        print(f"Loading model for {name}...")
        
        with open(paths["features_path"], 'r') as f:
            features = json.load(f)
        
        mock_env = DummyVecEnv([lambda: MockEnv(features)])
        
        model = PPO.load(paths["model_path"])
        vec_normalize = VecNormalize.load(paths["vec_norm_path"], mock_env)
        
        agent_models[name] = {'model': model, 'vec_norm': vec_normalize, 'features': features}

    all_strategies = list(agents_to_load.keys()) + ['Static (Ratio=1.0)', 'No Hedge (Ratio=0.0)']

    # --- 2. Initialize Simulation State ---
    simulation_results = {name: {'pnls': [], 'costs': [], 'hedge_ratios': [], 'entropies': []} for name in all_strategies}
    current_hedge_ratios = {name: 0.0 for name in all_strategies}
    transaction_cost = 0.0005

    # --- 3. Run Day-by-Day Simulation Loop ---
    eval_period_len = len(df_test) - 1
    print(f"\n--- Running simulation for {eval_period_len} time steps ---")

    for i in range(eval_period_len):
        spot_return = df_test['spot_ret'].iloc[i]
        futures_return = df_test['fut_ret'].iloc[i]

        # --- Evaluate RL Agents ---
        for name, components in agent_models.items():
            model = components['model']
            vec_norm = components['vec_norm']
            features = components['features']
            
            # a. Construct observation
            market_features = df_test[features].iloc[i].values
            current_hr = current_hedge_ratios[name]
            obs = np.concatenate([market_features, [current_hr]]).astype(np.float32)

            # b. Manually normalize the observation
            normalized_obs = vec_norm.normalize_obs(obs)
            
            # c. Get prediction
            obs_tensor = th.as_tensor(normalized_obs, device=model.device).reshape(1, -1)
            distribution = model.policy.get_distribution(obs_tensor)
            action = distribution.mode()
            entropy = distribution.entropy()
            
            # d. Clip action
            new_hedge_ratio = th.clamp(action, 0.0, 1.5).detach().cpu().numpy().flatten()[0]
            
            # e. Calculate PnL and Cost for this step
            pnl = spot_return - current_hr * futures_return
            cost = transaction_cost * abs(new_hedge_ratio - current_hr)
            
            # f. Store results
            simulation_results[name]['pnls'].append(pnl)
            simulation_results[name]['costs'].append(cost)
            simulation_results[name]['hedge_ratios'].append(new_hedge_ratio)
            simulation_results[name]['entropies'].append(entropy.detach().cpu().numpy().flatten()[0])
            
            # g. Update state for next iteration
            current_hedge_ratios[name] = new_hedge_ratio

        # --- Evaluate Baselines ---
        # No Hedge
        pnl_no_hedge = spot_return
        simulation_results['No Hedge (Ratio=0.0)']['pnls'].append(pnl_no_hedge)
        simulation_results['No Hedge (Ratio=0.0)']['costs'].append(0)
        simulation_results['No Hedge (Ratio=0.0)']['hedge_ratios'].append(0)
        simulation_results['No Hedge (Ratio=0.0)']['entropies'].append(0)

        # Static Hedge
        current_hr_static = current_hedge_ratios['Static (Ratio=1.0)']
        pnl_static = spot_return - current_hr_static * futures_return
        cost_static = transaction_cost * abs(1.0 - current_hr_static) if i == 0 else 0
        simulation_results['Static (Ratio=1.0)']['pnls'].append(pnl_static)
        simulation_results['Static (Ratio=1.0)']['costs'].append(cost_static)
        simulation_results['Static (Ratio=1.0)']['hedge_ratios'].append(1.0)
        simulation_results['Static (Ratio=1.0)']['entropies'].append(0)
        current_hedge_ratios['Static (Ratio=1.0)'] = 1.0


    # --- 4. Calculate and Print Metrics ---
    print("\n--- Comparative Performance Metrics ---")
    for name, results in simulation_results.items():
        metrics = calculate_metrics(results)
        print(f"\n--- {name} ---")
        for key, value in metrics.items():
            print(f"{key:<15}: {value: .5f}")

    # --- 5. Plotting ---
    print("\n--- Generating Comparison Plots ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(4, 1, figsize=(18, 32), sharex=True)
    colors = {'Cost Control': 'red', 'Profit Maximization': 'blue', 'Risk Reduction': 'green', 'Static (Ratio=1.0)': 'black', 'No Hedge (Ratio=0.0)': 'orange'}

    # Plot 1: Cumulative Net PnL
    for name, results in simulation_results.items():
        net_pnl = np.cumsum(np.array(results['pnls']) - np.array(results['costs']))
        axs[0].plot(net_pnl, label=name, color=colors.get(name, 'gray'))
    axs[0].set_title('Comparative Net PnL (PnL - Costs)', fontsize=16)
    axs[0].set_ylabel('Cumulative Net PnL', fontsize=12)
    axs[0].legend()
    axs[0].axhline(0, color='black', linestyle='--', linewidth=0.8)

    # Plot 2: Cumulative Costs
    for name, results in simulation_results.items():
        axs[1].plot(np.cumsum(results['costs']), label=name, color=colors.get(name, 'gray'))
    axs[1].set_title('Comparative Cumulative Transaction Costs', fontsize=16)
    axs[1].set_ylabel('Cumulative Costs', fontsize=12)
    axs[1].legend()

    # Plot 3: Hedge Ratio Policies
    for name, results in simulation_results.items():
        axs[2].plot(results['hedge_ratios'], label=name, color=colors.get(name, 'gray'), alpha=0.8, drawstyle='steps-post')
    axs[2].set_title('Comparative Hedge Ratio Policies', fontsize=16)
    axs[2].set_ylabel('Hedge Ratio', fontsize=12)
    axs[2].legend()

    # Plot 4: Policy Uncertainty (Entropy)
    for name, results in simulation_results.items():
        axs[3].plot(results['entropies'], label=name, color=colors.get(name, 'gray'), alpha=0.8)
    axs[3].set_title('Comparative Policy Uncertainty (Entropy)', fontsize=16)
    axs[3].set_ylabel('Entropy', fontsize=12)
    axs[3].set_xlabel('Time Step in Test Set', fontsize=12)
    axs[3].legend()

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'v3/logs/realtime_simulation_results_{timestamp}.png'
    plt.savefig(save_path)
    print(f"\n--- Simulation complete. Plot saved to {save_path} ---")
    plt.show()

if __name__ == "__main__":
    main()
