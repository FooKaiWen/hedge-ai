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

---

## Phase 1: Variance Minimization - Results

The **Primary Approach (Variance Minimization)** was implemented and tested. The agent was trained to maximize the negative squared hedged return.

### Final Metrics:

| Strategy        |          VRE |     Variance |    Sharpe Ratio |   Cumulative PnL |
|:----------------|-------------:|-------------:|----------------:|-----------------:|
| RL Agent        |      -0.0245 |     0.000009 |         -0.4463 |          -0.1198 |
| No Hedge        |       0.0000 |     0.000008 |         -0.3121 |          -0.0828 |
| Full Hedge      |     -25.2882 |     0.000222 |         -0.1907 |          -0.2594 |
| OLS Hedge       |       0.0002 |     0.000008 |         -0.3270 |          -0.0867 |

### Findings:

The experiment was **not successful**.

1.  **Failure to Reduce Risk:** The agent's primary goal was to reduce variance. However, it achieved a negative VRE (`-0.0245`), meaning it slightly increased risk compared to taking no hedge at all. It was significantly outperformed by the simple OLS/MVHR statistical model, which achieved the lowest variance.
2.  **Poor Performance:** The agent also failed on secondary metrics, delivering a worse PnL and Sharpe Ratio than both the No Hedge and OLS Hedge strategies.
3.  **Strategic Insight:** A pure variance-minimization reward signal (`-pnl^2`) appears to be insufficient or too noisy for the agent to learn an effective policy. It struggles to beat a simple, global statistical benchmark like OLS.

---

## Phase 2: Utility Maximization - Next Steps

Based on the failure of the pure variance minimization approach, the next logical step is to implement the **Secondary Approach (Utility Maximization)** outlined in the initial plan. This provides a more balanced and potentially more stable reward signal for the agent to learn from.

### 1. Implement Utility Function Reward

- **Action:** Modify the `step` function in `v3/main_risk_reduction.py` to use the utility function.
- **New Reward Formula:** `reward = pnl - (0.5 * risk_aversion * pnl**2)`
- **Details:**
    - A `risk_aversion` parameter (e.g., starting at `1.0`) will be added to the environment's `__init__` method.
    - This formula explicitly tells the agent to find a balance between making a profit (`pnl`) and avoiding large swings in returns (the `pnl**2` penalty). This is a more nuanced objective and is a standard in modern portfolio theory.

### 2. Re-Train and Evaluate

- **Action:** Execute the modified `main_risk_reduction.py` script to train a new agent based on this utility function.
- **Details:** The existing training pipeline, including hyperparameters and timesteps, will be used for the initial run.

### 3. Compare and Analyze

- **Action:** Execute the `compare_risk_reduction.py` script on the new model.
- **Objective:** To determine if the utility-based agent can successfully reduce variance (achieve a positive VRE) while maintaining a reasonable Sharpe Ratio, and to see if it can finally outperform the OLS/MVHR benchmark.
