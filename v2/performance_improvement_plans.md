# RL Agent Performance Improvement Plans

This document outlines three distinct strategic plans to improve the performance of the PPO-based hedging agent. The initial baseline showed that the agent underperformed compared to a simple OLS/MVHR benchmark. These plans are designed to address common challenges in RL and systematically enhance the agent's effectiveness.

---

### Plan A: Reward Shaping and Increased Training (DONE)

**Objective:** To provide the agent with a more immediate and consistent learning signal, and to allow more time for convergence. The current reward is sparse (given only at the end of an episode), which can make learning difficult.

**Strategy:**
1.  **Implement Dense Rewards:** Modify the environment to provide a reward at every single timestep. A common and effective approach for volatility-minimization tasks is to use the negative squared hedged return as the reward for each step.
2.  **Increase Training Duration:** A more complex reward signal and environment may require more training steps for the agent to learn an effective policy.

**Execution Steps:**
1.  **Modify `HedgingEnv`:**
    -   In the `step` method, change the reward calculation. Instead of accumulating returns and calculating variance at the end, calculate the reward at each step: `reward = -(hedged_r ** 2)`.
    -   Remove the terminal reward calculation (`reward = -np.var(self.hedged_returns)`).
2.  **Extend Training:**
    -   In the `if __name__ == "__main__"` block, increase the `total_timesteps` argument in the `model.learn()` call from `100000` to `300000`.
3.  **Execute and Analyze:**
    -   Run the `main.py` script to train the new agent.
    -   Compare the resulting variance and Variance Reduction Effectiveness (VRE) against the baseline and benchmark models.

---

### Plan B: Hyperparameter Optimization

**Objective:** To find the optimal set of hyperparameters for the PPO algorithm. The default parameters used in the baseline are rarely optimal for a specific problem.

**Strategy:**
1.  **Automated Tuning:** Use a library like Optuna to systematically search for the best-performing hyperparameters.
2.  **Define Search Space:** Select a range of sensible values for key PPO parameters, such as the learning rate, `n_steps`, `gamma` (discount factor), and `ent_coef` (entropy coefficient).
3.  **Retrain with Best Parameters:** Once the optimization study is complete, use the best-found hyperparameters to train a new agent for a full duration.

**Execution Steps:**
1.  **Create an Optimization Script:**
    -   Write a new Python script (`tune_hyperparameters.py`) that imports `optuna` and `main.py`'s functions.
    -   Define an `objective` function that takes an `optuna.trial` object.
    -   Inside the function, use `trial.suggest_...` to sample hyperparameters.
    -   Train and evaluate an agent within the objective function, returning the final variance or VRE as the metric for Optuna to optimize.
2.  **Run the Study:**
    -   Execute the optimization script and let it run for a predefined number of trials (e.g., 50-100).
3.  **Train Final Model:**
    -   Hard-code the best hyperparameters found by Optuna into the `PPO` constructor in `main.py`.
    -   Run `main.py` one last time to generate the final, optimized model and its performance results.

---

### Plan C: Advanced Algorithm and Feature Engineering

**Objective:** To explore whether a different, more sample-efficient RL algorithm can achieve better performance and to refine the information provided to the agent.

**Strategy:**
1.  **Switch to Soft Actor-Critic (SAC):** SAC is another powerful algorithm for continuous action spaces that often outperforms PPO in terms of sample efficiency and final performance.
2.  **Refine Feature Set:** The current feature set is large. It's possible that some features are irrelevant or noisy, which can confuse the agent. Experimenting with a smaller, more curated set of features can lead to better results.

**Execution Steps:**
1.  **Modify `train_agent`:**
    -   Import `SAC` from `stable_baselines3` instead of `PPO`.
    -   Instantiate the `SAC` model, keeping most of the training infrastructure the same.
2.  **Modify `HedgingEnv`:**
    -   Change the `self.features` list to a more focused set. For example, a set focused purely on market dynamics:
        ```python
        self.features = [
            'spot_ret_lag', 'fut_ret_lag', 'spot_std_20', 'fut_std_20',
            'basis', 'volume_z', 'hr_ols_60'
        ]
        ```
3.  **Execute and Analyze:**
    -   Run the modified `main.py` script.
    -   Compare the performance of the SAC agent with the refined feature set against the original PPO baseline and the OLS/MVHR benchmark.
