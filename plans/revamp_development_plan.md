# Development Plan: Revamping the Hedging Agent

This document outlines a plan to diagnose and address the issue of the reinforcement learning agents failing to learn a meaningful hedging strategy (i.e., consistently choosing a hedge ratio of 0). The focus is on a fundamental redesign of the learning environment and reward structure to better align with the principles of financial hedging.

## Phase 1: Foundational Analysis & Environment Redesign

**Objective:** To create a learning environment that accurately represents the hedging problem and provides clear, meaningful rewards that incentivize risk management over short-term profit.

*   **Task 1.1: Deepen Data Analysis**
    *   **Step 1.1.1:** Examine the features in `data/fcpo_daily.csv` to identify additional predictive signals for the agent. This includes analyzing trading volume, open interest, and other market indicators that might provide context for price movements.

*   **Task 1.2: Enhance the State (Observation) Space**
    *   **Step 1.2.1:** Enrich the agent's observation with more contextual features to improve its decision-making capability.
    *   **Step 1.2.2:** Add a rolling measure of historical price volatility (e.g., 30-day standard deviation of returns) to inform the agent about the current risk environment.
    *   **Step 1.2.3:** Include time-based features, such as the day of the week, day of the month, and time remaining until the futures contract expires.
    *   **Step 1.2.4:** Add the current CPO inventory as a percentage of the expected production to help the agent understand its exposure.

*   **Task 1.3: Redesign the Reward Function**
    *   **Step 1.3.1:** Replace the single-step PnL reward with a more sophisticated function that reflects the true goal of hedging.
    *   **Step 1.3.2:** Implement a reward based on the **Sharpe Ratio calculated over a rolling window** (e.g., the last 30 days). This will encourage the agent to prioritize stable, risk-adjusted returns over volatile, short-term gains.
    *   **Step 1.3.3:** The reward at each step will be the change in the rolling Sharpe Ratio, rewarding the agent for actions that improve its risk-adjusted performance over time.

*   **Task 1.4: Introduce a "No-Hedge" Benchmark**
    *   **Step 1.4.1:** Modify the `HedgeEnv` to simultaneously run a parallel simulation of a baseline "no-hedging" strategy.
    *   **Step 1.4.2:** Calculate the agent's performance (e.g., portfolio value) relative to this benchmark.
    *   **Step 1.4.3:** The reward could be structured as the agent's return minus the benchmark's return, directly incentivizing the agent to provide value above a passive approach.

## Phase 2: Advanced Agent Training & Evaluation

**Objective:** To improve the training methodology and evaluation criteria to better guide the agent and assess its performance.

*   **Task 2.1: Implement Curriculum Learning**
    *   **Step 2.1.1:** Design a curriculum to make the problem easier for the agent to learn initially and then gradually increase the difficulty.
    *   **Step 2.1.2:** Start by training the agent in a simplified environment (e.g., with zero transaction costs and deterministic production).
    *   **Step 2.1.3:** Gradually introduce more complexity, such as transaction costs, stochastic production, and market volatility.

*   **Task 2.2: Refine Evaluation Metrics**
    *   **Step 2.2.1:** Expand the evaluation metrics beyond ROI and Sharpe Ratio to include more sophisticated risk measures.
    *   **Step 2.2.2:** Implement metrics such as **Value at Risk (VaR)**, **Conditional Value at Risk (CVaR)**, and the **Sortino Ratio** (which focuses on downside volatility).
    *   **Step 2.2.3:** These metrics will provide a more complete picture of the agent's ability to manage risk.

## Phase 3: Implementation, Validation, and Iteration

**Objective:** To implement the proposed changes, run new experiments, and validate the performance of the revamped agent.

*   **Task 3.1: Code Implementation**
    *   **Step 3.1.1:** Refactor `src/environment/hedge_env.py` to implement the enhanced state space and the new rolling Sharpe Ratio reward function.
    *   **Step 3.1.2:** Update `src/agents/evaluate_agent.py` to incorporate the curriculum learning strategy and the refined evaluation metrics.

*   **Task 3.2: Run New Experiments**
    *   **Step 3.2.1:** Retrain the RL agents (initially focusing on the 'risk' or a new 'benchmark-beating' agent type) using the revamped environment.
    *   **Step 3.2.2:** Systematically log all experiment parameters, results, and evaluation metrics.

*   **Task 3.3: Analyze and Compare**
    *   **Step 3.3.1:** Rigorously analyze the results of the new experiments.
    *   **Step 3.3.2:** Specifically, analyze the distribution of `hedge_ratio` values to confirm that the agent is no longer stuck at zero.
    *   **Step 3.3.3:** Compare the risk-adjusted returns and other performance metrics of the new agent against the previous results and the "no-hedge" benchmark.
