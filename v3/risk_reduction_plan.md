# Plan: Developing a Risk-Reduction RL Agent

This document outlines the strategic plan to develop, train, and evaluate a new reinforcement learning agent specifically designed for risk reduction in a financial hedging context.

### 1. Foundational Script & Environment

- **Action:** Create a new Python script named `v3/main_risk_reduction.py`.
- **Details:** This script will be adapted from `v3/main.py`. It will serve as the primary training and evaluation pipeline for the new agent, ensuring its development is separate from the existing profit-maximization agent.

### 2. Core Strategy: Re-engineering the Reward Function

The agent's behavior is dictated by its reward. To shift its focus from maximizing profit to minimizing risk, the reward mechanism will be fundamentally altered.

- **Action:** Modify the `step` method within the `HedgeEnvV3` class in the new script.
- **Details:** Two reward structures will be implemented and tested.

    1.  **Primary Approach (Variance Minimization):**
        -   **New Reward Formula:** `reward = -((spot_return - hedge_ratio * futures_return) ** 2)`
        -   **Objective:** This reward is the negative squared hedged return. By learning to maximize this value, the agent is mathematically driven to minimize the variance of its returns. This directly aligns the agent's goal with our primary objective of risk reduction. Transaction costs will be omitted from this reward to ensure the agent's focus remains purely on achieving stability.

    2.  **Secondary Approach (Utility Maximization):**
        -   **New Reward Formula:** `reward = pnl - (0.5 * risk_aversion * pnl**2)`
        -   **Objective:** This implements a classic financial utility function. It creates a trade-off between generating profit (`pnl`) and controlling risk (the `pnl**2` term). A `risk_aversion` parameter will be introduced (defaulting to `1.0`) to allow for tuning how heavily the agent penalizes volatility. This provides a more nuanced objective than pure variance minimization.

### 3. Evaluation Framework

A new set of metrics is required to accurately assess the performance of a risk-reduction strategy.

- **Action:** The evaluation section of the script will be updated to calculate and report on risk-centric metrics.
- **Primary Metrics:**
    -   **Variance & Standard Deviation:** To measure the absolute volatility of the hedged portfolio. A lower value is better.
    -   **Variance Reduction Effectiveness (VRE):** This will be the main success indicator. It is calculated as `1 - (Variance_Hedged / Variance_Unhedged)`. A high positive VRE demonstrates effective risk reduction.
- **Secondary Metric:**
    -   **Sharpe Ratio:** To monitor risk-adjusted returns and ensure that the reduction in variance does not lead to an unacceptable loss of profit.

### 4. Benchmarking and Comparison

To prove the agent's effectiveness, it must be compared against established strategies.

- **Action:** A new comparison script, `v3/compare_risk_reduction.py`, will be created.
- **Details:** This script will:
    1.  Load the trained risk-reduction model.
    2.  Evaluate it against the standard benchmarks: No Hedge, Full Hedge, and OLS/MVHR.
    3.  Generate a performance summary table and comparison plots that prominently feature Variance and VRE as the key performance indicators.

### 5. Artifact Management

Clear and distinct naming conventions will be used to manage all files associated with this new agent.

- **Action:** All output files will be named to reflect their risk-reduction purpose.
- **Examples:**
    -   Model file: `ppo_hedge_risk_reduction.zip`
    -   Normalization stats: `vec_normalize_risk_reduction.pkl`
    -   Feature list: `features_risk_reduction.json`
    -   Evaluation plot: `v3_risk_reduction_comparison.png`

This detailed plan ensures a structured and methodologically sound approach to developing an RL agent that is an expert in risk reduction.
