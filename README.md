# Hedge-AI: CPO Hedging Strategy Optimization

This project develops an intelligent agent using Reinforcement Learning (RL) to find optimal hedging strategies for Crude Palm Oil (CPO) production, aiming to maximize profit while managing risk.

## Project Structure

```
hedge-ai/
├── data/                # Processed data used for training and evaluation
├── models/              # Trained models and hyperparameters
├── notebooks/           # Jupyter notebooks for data exploration and processing
├── outputs/             # Evaluation results (CSVs and plots)
├── raw_data/            # Raw data files
├── src/                 # Source code
│   ├── agents/          # RL agent tuning and evaluation scripts
│   ├── data_processing/ # Data processing scripts
│   └── environment/     # Custom Gym environment for hedging
├── poetry.lock          # Poetry lock file
├── pyproject.toml       # Project dependencies
└── README.md            # This file
```

## Getting Started

### 1. Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

To install the required packages, run:
```bash
poetry install
```
This will create a virtual environment and install all necessary libraries.

### 2. Data Processing

The data processing pipeline involves several steps. The scripts for these are located in the `src/data_processing/` directory. You should run them in the following order:

*(Note: The original data fetching and processing was done in notebooks, which have been converted to `.py` scripts. The following demonstrates how to run them.)*

1.  **Fetch Initial Data:**
    ```bash
    poetry run python src/data_processing/fetch.py
    ```
2.  **Process FFB Data:**
    ```bash
    poetry run python src/data_processing/process_ffb.py
    ```
3.  **Process CPO/OER Data:**
    ```bash
    poetry run python src/data_processing/process_cpo_oer.py
    ```
4.  **Merge Datasets:**
    ```bash
    poetry run python src/data_processing/merge_datasets.py
    ```
After these steps, you should have a `fcpo_daily.csv` file in the `data/` directory.

### 3. Agent Training and Evaluation

The core of the project is to train and evaluate agents with different objectives.

#### Agent Types

There are three types of agents you can train, each with a different reward strategy:
*   **`profit`**: Aims to maximize the total profit.
*   **`risk`**: Aims to maximize risk-adjusted returns (using a Sharpe-like reward).
*   **`cost`**: Aims to minimize the costs associated with hedging (transaction costs, etc.).

#### Step 1: Hyperparameter Tuning

Before training a final agent, it's crucial to find the best hyperparameters. Use the `tune_agent.py` script for this. The script uses Optuna to perform the search and saves the best parameters to the `models/` directory.

**Usage:**
```bash
poetry run python src/agents/tune_agent.py --agent <agent_type>
```

**Examples:**

*   Tune the profit-seeking agent:
    ```bash
    poetry run python src/agents/tune_agent.py --agent profit
    ```
*   Tune the risk-averse agent:
    ```bash
    poetry run python src/agents/tune_agent.py --agent risk
    ```
*   Tune the cost-minimizing agent:
    ```bash
    poetry run python src/agents/tune_agent.py --agent cost
    ```
This will create a file like `models/profit_agent_best_hyperparameters.pkl`.

#### Step 2: Evaluating the Agent

Once you have the best hyperparameters, you can train an agent for a longer duration and evaluate its performance on a test period. Use the `evaluate_agent.py` script for this.

This script will:
1.  Load the best hyperparameters for the specified agent.
2.  Train a PPO agent.
3.  Save the trained agent to the `models/` directory (e.g., `models/evaluated_profit_agent_...zip`).
4.  Run an evaluation on the test data.
5.  Save the evaluation results as a CSV and a performance plot in the `outputs/` directory.

**Usage:**
```bash
poetry run python src/agents/evaluate_agent.py --agent <agent_type> --timesteps <number_of_timesteps>
```

**Example:**

*   Train and evaluate the `risk` agent for 200,000 timesteps:
    ```bash
    poetry run python src/agents/evaluate_agent.py --agent risk --timesteps 200000
    ```

This provides a full workflow from data processing to agent evaluation.