import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_latest_result(agent_type: str):
    """Load the most recent evaluation result file for a given agent."""
    pattern = f"outputs/{agent_type}_evaluation_results_*.csv"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No evaluation results found for {agent_type}. Run evaluate_agents.py first.")
    latest_file = max(files, key=os.path.getctime)
    print(f"Loaded latest results for {agent_type}: {latest_file}")
    return pd.read_csv(latest_file)

# def compute_metrics_old(df: pd.DataFrame, initial_cash: float = 1_000_000):
#     """Compute ROI, Sharpe Ratio, and Cost Efficiency Ratio for a results dataframe."""
#     portfolio_values = df["portfolio_value"].values
#     rewards = df["reward"].values
#     returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
#     # ROI
#     roi = (portfolio_values[-1] - initial_cash) / initial_cash

#     # Sharpe Ratio (risk-adjusted return)
#     mean_ret = np.mean(returns)
#     std_ret = np.std(returns) + 1e-8  # avoid divide-by-zero
#     sharpe_ratio = mean_ret / std_ret

#     # Cost Efficiency Ratio (net gain per total cost)
#     total_cost_proxy = np.abs(rewards[rewards < 0]).sum() + 1e-8
#     cost_efficiency = (portfolio_values[-1] - initial_cash) / total_cost_proxy

#     return {
#         "Final Portfolio": portfolio_values[-1],
#         "ROI": roi,
#         "Sharpe Ratio": sharpe_ratio,
#         "Cost Efficiency": cost_efficiency
#     }

def compute_metrics(df: pd.DataFrame, initial_cash: float = 1_000_000):
    """Compute ROI, Sharpe Ratio, and Cost Efficiency Ratio based on consistent definitions."""
    portfolio_values = df["portfolio_value"].values
    if len(portfolio_values) < 2:
        return {"Final Portfolio": initial_cash, "ROI": 0.0, "Sharpe Ratio": 0.0, "Cost Efficiency": 0.0}

    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # ROI
    roi = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

    # Sharpe Ratio
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)

    # Cost Efficiency Ratio (net gain per total transaction cost)
    if "transaction_cost" in df.columns:
        total_cost = df["transaction_cost"].sum()
    else:
        # fallback proxy if not recorded
        total_cost = np.abs(df["reward"][df["reward"] < 0]).sum()
    cost_efficiency = ((portfolio_values[-1] - portfolio_values[0]) / (total_cost + 1e-8))

    return {
        "Final Portfolio": portfolio_values[-1],
        "ROI": float(np.nan_to_num(roi)),
        "Sharpe Ratio": float(np.nan_to_num(sharpe_ratio)),
        "Cost Efficiency": float(np.nan_to_num(cost_efficiency))
    }


def main():
    agent_types = ["profit", "risk", "cost"]
    results_summary = []

    for agent in agent_types:
        df = load_latest_result(agent)
        metrics = compute_metrics(df)
        metrics["Agent"] = agent.capitalize()
        results_summary.append(metrics)

    summary_df = pd.DataFrame(results_summary)[
        ["Agent", "Final Portfolio", "ROI", "Sharpe Ratio", "Cost Efficiency"]
    ]
    print("\n=== 📊 Agent Performance Summary ===")
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    x = np.arange(len(agent_types))

    plt.bar(x - bar_width, summary_df["ROI"], width=bar_width, label="ROI")
    plt.bar(x, summary_df["Sharpe Ratio"], width=bar_width, label="Sharpe Ratio")
    plt.bar(x + bar_width, summary_df["Cost Efficiency"], width=bar_width, label="Cost Efficiency")

    plt.xticks(x, summary_df["Agent"])
    plt.ylabel("Metric Value (normalized scale)")
    plt.title("Agent Comparison: ROI vs Sharpe vs Cost Efficiency")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    output_plot = "outputs/agent_comparison_metrics.png"
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"\nComparison plot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    main()
