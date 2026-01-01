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
import logging
import time
from google import genai

MODEL="gemini-2.5-flash"

# --- LLM Integration Setup ---
try:
    # Get the API key from an environment variable
    api_key = "AIzaSyBoX8ususrzHouOdjR4Nx5G7aYDF5960vw" # os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise KeyError("GOOGLE_API_KEY not found")
    LLM_MODEL = genai.Client(api_key=api_key)
    print("Successfully configured Gemini API.")
except (AttributeError, KeyError):
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    print("Please set the GOOGLE_API_KEY to your Gemini API key to run this script.")
    exit()

# Create the generative model
# --- End LLM Integration Setup ---


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
# 2. AGENTIC DSS - LLM MODULES
# ==============================================================================
def get_agent_proposal_with_reasoning(agent_name, hedge_ratio, market_view, logger):
    """
    Generates a rationale for an agent's proposed hedge ratio using a real LLM.
    """
    logger.debug(f"Generating proposal for {agent_name} with initial ratio {hedge_ratio:.2f}")
    prompt = f"""
    You are an expert financial agent with the persona of a '{agent_name}'.
    Your task is to provide a brief, one-sentence rationale for a hedging decision.
    
    Current Market View:
    - CPO Spot Price: {market_view.get('spot_close', 'N/A'):.2f}
    - FCPO Futures Price: {market_view.get('fut_close', 'N/A'):.2f}
    - FCPO Next Day Prediction: {market_view.get('next_day_pred', 'N/A'):.2f}
    - FCPO Next Month Prediction: {market_view.get('next_month_pred', 'N/A'):.2f}
    - 20-day Spot Volatility (Std Dev): {market_view.get('spot_std_20', 'N/A'):.2f}
    - Volatility Regime: {market_view.get('vol_regime', 'N/A')}
    - Basis Percentage (Spot-Futures): {market_view.get('basis_pct', 'N/A'):.2f}%
    
    Based on your persona, you have decided on a hedge ratio of: {hedge_ratio:.2f}
    
    Provide a concise, single-sentence justification for this decision, starting with 'As the {agent_name}, ...'.
    
    Example for Profit Maximization: "As the Profit Maximization agent, I recommend an aggressive hedge to capitalize on expected price movements."
    Example for Risk Reduction: "As the Risk Reduction agent, the high volatility necessitates a strong hedge to protect our position."
    Example for Cost Control: "As the Cost Control agent, I advise a minimal change to our current hedge to reduce transaction costs."

    Justification:
    """
    logger.debug(f"Prompt for {agent_name}:\n{prompt}")
    try:
        response = LLM_MODEL.models.generate_content(model=MODEL, contents=prompt)
        time.sleep(1) # API call delay
        rationale = response.text.strip().replace('\n', ' ')
        logger.debug(f"LLM Rationale for {agent_name}: '{rationale}'")
    except Exception as e:
        logger.error(f"Error calling LLM for {agent_name}: {e}")
        # Fallback to mock rationale
        rationale = f"As the {agent_name}, I propose a hedge ratio of {hedge_ratio:.2f} based on my internal analysis (LLM fallback)."
        logger.warning(f"Using fallback rationale for {agent_name}.")

    return {"ratio": hedge_ratio, "rationale": rationale}


def moderate_debate_round(proposals, market_view, current_hedge, round_number, logger):
    """
    Simulates the CIO's role as a debate moderator for a single round.
    """
    logger.debug(f"--- Moderating Debate Round {round_number} ---")
    prompt = f"""
    You are a Chief Investment Officer (CIO) moderating a debate between three specialist AI agents. This is round {round_number} of the debate.

    Current Market View:
    - CPO Spot Price: {market_view.get('spot_close', 'N/A'):.2f}
    - FCPO Futures Price: {market_view.get('fut_close', 'N/A'):.2f}
    - FCPO Next Day Prediction: {market_view.get('next_day_pred', 'N/A'):.2f}
    - FCPO Next Month Prediction: {market_view.get('next_month_pred', 'N/A'):.2f}
    - 20-day Spot Volatility (Std Dev): {market_view.get('spot_std_20', 'N/A'):.2f}
    - Volatility Regime: {market_view.get('vol_regime', 'N/A')}
    - Basis Percentage (Spot-Futures): {market_view.get('basis_pct', 'N/A'):.2f}%
    - Current Hedge Ratio: {current_hedge:.2f}

    Current Agent Proposals:
    1. Profit Maximization: Ratio={proposals['Profit Maximization']['ratio']:.2f}, Rationale="{proposals['Profit Maximization']['rationale']}"
    2. Risk Reduction: Ratio={proposals['Risk Reduction']['ratio']:.2f}, Rationale="{proposals['Risk Reduction']['rationale']}"
    3. Cost Control: Ratio={proposals['Cost Control']['ratio']:.2f}, Rationale="{proposals['Cost Control']['rationale']}"

    **Your Task**:
    1.  **Critique**: Write a brief, overall critique of the agents' current positions. Are they too far apart? Is one agent ignoring a key market signal?
    2.  **Propose Compromise**: Suggest a single "compromise" hedge ratio that balances the current views.
    3.  **Ask Questions**: Formulate a specific, challenging question for EACH of the three agents to force them to reconsider their stance in light of your critique and compromise.

    **Output Format**:
    Your entire response MUST be a JSON object.
    
    Example Output:
    {{
      "critique": "The Profit and Risk agents are too far apart, with one ignoring the clear market volatility. The Cost agent is being overly cautious.",
      "compromise_ratio": 0.85,
      "questions_for_agents": {{
        "Profit Maximization": "Can you justify your high-risk stance given the 20-day volatility is high?",
        "Risk Reduction": "Is a full hedge necessary, or can we capture some upside while still managing risk?",
        "Cost Control": "Are you prioritizing cost-saving to the detriment of protecting the firm from a clear market threat?"
      }}
    }}
    """
    logger.debug(f"CIO Moderator Prompt (Round {round_number}):\n{prompt}")
    response_text = ""
    try:
        response = LLM_MODEL.models.generate_content(model=MODEL, contents=prompt)
        time.sleep(1) # API call delay
        response_text = response.text
        cleaned_text = response_text.strip().replace("```json", "").replace("```", "")
        feedback = json.loads(cleaned_text)
        logger.debug(f"CIO Moderator Feedback (Round {round_number}): {json.dumps(feedback, indent=2)}")
        return feedback
    except Exception as e:
        logger.error(f"Error during CIO moderation: {e}")
        logger.error(f"LLM raw response was: {response_text}")
        return None # Signal an error


def get_agent_rebuttal(agent_name, current_proposal, cio_feedback, market_view, logger):
    """
    Simulates an agent's rebuttal round using an LLM.
    The agent re-evaluates its position based on the CIO's feedback.
    """
    logger.debug(f"Getting rebuttal from {agent_name}. Current proposal: {current_proposal['ratio']:.2f}")
    prompt = f"""
    You are an expert financial agent with the persona of a '{agent_name}'.
    You are in a debate with a CIO and other agents to decide on a hedge ratio.

    Your previous proposal was:
    - Ratio: {current_proposal['ratio']:.2f}
    - Rationale: "{current_proposal['rationale']}"

    The CIO has reviewed the proposals and provided the following feedback and questions for you:
    - CIO's Critique: "{cio_feedback['critique']}"
    - CIO's Proposed Compromise: {cio_feedback['compromise_ratio']:.2f}
    - Question for you: "{cio_feedback['questions_for_agents'][agent_name]}"
    
    Current Market View:
    - CPO Spot Price: {market_view.get('spot_close', 'N/A'):.2f}
    - FCPO Futures Price: {market_view.get('fut_close', 'N/A'):.2f}
    - FCPO Next Day Prediction: {market_view.get('next_day_pred', 'N/A'):.2f}
    - FCPO Next Month Prediction: {market_view.get('next_month_pred', 'N/A'):.2f}
    - 20-day Spot Volatility (Std Dev): {market_view.get('spot_std_20', 'N/A'):.2f}
    - Volatility Regime: {market_view.get('vol_regime', 'N/A')}
    - Basis Percentage (Spot-Futures): {market_view.get('basis_pct', 'N/A'):.2f}%

    **Your Task**:
    1. Re-evaluate your position based on the CIO's feedback and question.
    2. Decide on a new hedge ratio. You can stick to your old one or propose a new one.
    3. Provide a new, concise rationale for your updated decision, explaining why you are changing or maintaining your stance.

    **Output Format**:
    Your entire response MUST be a JSON object with two keys: "new_ratio" (a float) and "new_rationale" (a string). Do not include any other text or formatting.

    Example Output:
    {{
      "new_ratio": 0.9,
      "new_rationale": "After considering the CIO's point about rising volatility, I am adjusting my proposal upward to 0.9 to better account for short-term risk, even if it slightly reduces our maximum profit potential."
    }}
    """
    logger.debug(f"Rebuttal prompt for {agent_name}:\n{prompt}")
    response_text = ""
    try:
        response = LLM_MODEL.models.generate_content(model=MODEL, contents=prompt)
        time.sleep(1) # API call delay
        response_text = response.text
        cleaned_text = response_text.strip().replace("```json", "").replace("```", "")
        decision = json.loads(cleaned_text)
        
        new_ratio = float(decision['new_ratio'])
        new_rationale = decision['new_rationale']

        logger.debug(f"{agent_name} Rebuttal: New Ratio={new_ratio:.2f}, Rationale='{new_rationale}'")
        return {"ratio": np.clip(new_ratio, 0.0, 1.5), "rationale": new_rationale}

    except Exception as e:
        logger.error(f"Error during rebuttal for {agent_name}: {e}")
        logger.error(f"LLM raw response was: {response_text}")
        logger.warning(f"Using fallback for {agent_name} rebuttal; sticking to original proposal.")
        return current_proposal


# Custom JSON Encoder for Numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def decide_converged_hedge_ratio(initial_proposals, market_view, current_hedge, logger, max_rounds=3, convergence_threshold=0.1):
    """
    Manages an iterative debate between agents, moderated by a CIO, to find a converged hedge ratio.
    """
    proposals = initial_proposals.copy()
    debate_history = []
    logger.debug(f"Initiating debate with initial proposals: {json.dumps(proposals, indent=2, cls=NpEncoder)}")

    for round_number in range(1, max_rounds + 1):
        logger.debug(f"--- Starting Debate Round {round_number} ---")
        
        ratios = [p['ratio'] for p in proposals.values()]
        std_dev = np.std(ratios)
        logger.debug(f"Proposal STD DEV: {std_dev:.4f} (Threshold: {convergence_threshold})")
        
        if std_dev < convergence_threshold:
            logger.info(f"Convergence met in round {round_number}.")
            final_ratio = np.mean(ratios)
            cio_rationale = f"Consensus reached after {round_number-1} rounds of debate. The agents converged on an average ratio of {final_ratio:.2f}."
            logger.debug(f"Final consensus rationale: {cio_rationale}")
            return np.clip(final_ratio, 0.0, 1.5), cio_rationale, debate_history

        cio_feedback = moderate_debate_round(proposals, market_view, current_hedge, round_number, logger)
        if cio_feedback is None:
            logger.error("CIO moderation failed. Breaking debate loop for executive decision.")
            break
        
        debate_history.append({"round": round_number, "cio_feedback": cio_feedback, "proposals_before_rebuttal": proposals.copy()})
        logger.debug(f"CIO Critique (Round {round_number}): {cio_feedback['critique']}")

        new_proposals = {}
        for agent_name in proposals.keys():
            new_proposals[agent_name] = get_agent_rebuttal(
                agent_name, 
                proposals[agent_name], 
                cio_feedback, 
                market_view,
                logger
            )
        proposals = new_proposals
        debate_history[-1]["proposals_after_rebuttal"] = proposals.copy()

    logger.warning("--- Max debate rounds reached or LLM error occurred. CIO making final executive decision. ---")
    
    prompt = f"""
    You are a Chief Investment Officer (CIO) making a final, executive decision after a multi-round debate with your AI agents failed to converge.

    Final Proposals:
    1. Profit Maximization: Ratio={proposals['Profit Maximization']['ratio']:.2f}, Rationale="{proposals['Profit Maximization']['rationale']}"
    2. Risk Reduction: Ratio={proposals['Risk Reduction']['ratio']:.2f}, Rationale="{proposals['Risk Reduction']['rationale']}"
    3. Cost Control: Ratio={proposals['Cost Control']['ratio']:.2f}, Rationale="{proposals['Cost Control']['rationale']}"

    **Your Task**:
    1.  Decide on a single, final hedge ratio between 0.0 and 1.5.
    2.  Provide a brief rationale for your final decision, explaining why you are overriding or siding with certain agents.
    **Output Format**:
    Your entire response MUST be a JSON object with two keys: "final_ratio" (a float) and "cio_rationale" (a string).
    """
    logger.debug(f"Executive Decision Prompt:\n{prompt}")
    response_text = ""
    try:
        response = LLM_MODEL.models.generate_content(model=MODEL, contents=prompt)
        time.sleep(1) # API call delay
        response_text = response.text
        cleaned_text = response_text.strip().replace("```json", "").replace("```", "")
        decision = json.loads(cleaned_text)
        final_ratio = float(decision['final_ratio'])
        cio_rationale = "[EXECUTIVE DECISION] " + decision['cio_rationale']
        logger.debug(f"CIO Executive Decision: {cio_rationale}")

    except Exception as e:
        logger.error(f"Error during final CIO decision: {e}")
        logger.error(f"LLM raw response was: {response_text}")
        final_ratio = np.mean([p['ratio'] for p in proposals.values()])
        cio_rationale = "[FALLBACK] Averaging final proposals due to LLM error."
        logger.warning(f"Using fallback executive decision. Average ratio: {final_ratio:.2f}")
    
    return np.clip(final_ratio, 0.0, 1.5), cio_rationale, debate_history


# ==============================================================================
# 3. METRICS CALCULATION
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
    net_pnl_steps = pnl - costs
    sharpe_ratio = (net_pnl_steps.mean() / (net_pnl_steps.std() + 1e-9)) * np.sqrt(252)
    
    # Portfolio Turnover
    turnover = np.sum(np.abs(np.diff(np.array([0.0] + list(hedge_ratios)))))
    
    return {
        'Total PnL': total_pnl,
        'Net PnL': net_pnl,
        'Total Costs': total_cost,
        'Sharpe Ratio': sharpe_ratio,
        'Turnover': turnover
    }

# ==============================================================================
# 4. MAIN SIMULATION SCRIPT
# ==============================================================================
def main():
    # --- Setup Detailed Logging ---
    log_dir = 'v3/logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f'simulation_log_{timestamp}.log')

    logger = logging.getLogger('AgenticDSSLogger')
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File handler for detailed DEBUG logs
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

        # Stream handler for INFO level logs on the console
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh_formatter = logging.Formatter('%(message)s')
        sh.setFormatter(sh_formatter)
        logger.addHandler(sh)
    # --- End Logging Setup ---
    
    logger.info("--- Starting Agentic DSS Simulation (with Iterative Debate and Detailed Logging) ---")
    
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
    
    class MockEnv(gym.Env):
        def __init__(self, features):
            super().__init__()
            self.action_space = spaces.Box(low=0.0, high=1.5, shape=(1,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(features) + 1,), dtype=np.float32)
        def step(self, action): pass
        def reset(self, *, seed=None, options=None): pass

    agent_models = {}
    for name, paths in agents_to_load.items():
        logger.info(f"Loading model for {name}...")
        with open(paths["features_path"], 'r') as f: features = json.load(f)
        mock_env = DummyVecEnv([lambda: MockEnv(features)])
        model = PPO.load(paths["model_path"])
        vec_normalize = VecNormalize.load(paths["vec_norm_path"], mock_env)
        agent_models[name] = {'model': model, 'vec_norm': vec_normalize, 'features': features}

    all_strategies = ['Agentic DSS'] + list(agents_to_load.keys()) + ['Static (Ratio=1.0)', 'No Hedge (Ratio=0.0)']

    # --- 2. Initialize Simulation State with Checkpoint Support ---
    checkpoint_path = os.path.join('v3/data', 'simulation_checkpoint.json')
    decision_history_path = os.path.join('v3/data', 'decision_history.json')
    
    # Check for existing checkpoint
    start_timestep = 0
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            start_timestep = checkpoint['last_completed_timestep'] + 1
            simulation_results = checkpoint['simulation_results']
            current_hedge_ratios = checkpoint['current_hedge_ratios']
            decision_history = checkpoint['decision_history']
            logger.info(f"\n✅ Checkpoint found! Resuming from timestep {start_timestep + 1}...")
            logger.info(f"   Already completed: {start_timestep} timesteps")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            start_timestep = 0
            simulation_results = {name: {'pnls': [], 'costs': [], 'hedge_ratios': [], 'entropies': []} for name in all_strategies}
            simulation_results['Agentic DSS']['rationales'] = []
            current_hedge_ratios = {name: 0.0 for name in all_strategies}
            decision_history = []
    else:
        simulation_results = {name: {'pnls': [], 'costs': [], 'hedge_ratios': [], 'entropies': []} for name in all_strategies}
        simulation_results['Agentic DSS']['rationales'] = []
        current_hedge_ratios = {name: 0.0 for name in all_strategies}
        decision_history = []
    
    transaction_cost = 0.0005

    # --- 3. Run Day-by-Day Simulation Loop ---
    eval_period_len = len(df_test) - 1
    remaining_steps = eval_period_len - start_timestep
    logger.info(f"\n--- Running simulation for {remaining_steps} remaining time steps (total: {eval_period_len}) ---")

    for i in range(start_timestep, eval_period_len):
        logger.info(f"\n{'='*20} Timestep {i+1}/{eval_period_len} {'='*20}")
        spot_return = df_test['spot_ret'].iloc[i]
        futures_return = df_test['fut_ret'].iloc[i]
        market_view = df_test.iloc[i]

        # --- Stage 1: Get Initial Proposals from each RL Agent ---
        agent_proposals = {}
        for name, components in agent_models.items():
            model, vec_norm, features = components['model'], components['vec_norm'], components['features']
            current_hr = current_hedge_ratios[name]
            
            obs = np.concatenate([df_test[features].iloc[i].values, [current_hr]]).astype(np.float32)
            normalized_obs = vec_norm.normalize_obs(obs)
            
            obs_tensor = th.as_tensor(normalized_obs, device=model.device).reshape(1, -1)
            distribution = model.policy.get_distribution(obs_tensor)
            action = distribution.mode()
            entropy = distribution.entropy().detach().cpu().numpy().flatten()[0]
            
            new_hedge_ratio = th.clamp(action, 0.0, 1.5).detach().cpu().numpy().flatten()[0]
            
            pnl = spot_return - current_hr * futures_return
            cost = transaction_cost * abs(new_hedge_ratio - current_hr)
            
            simulation_results[name]['pnls'].append(pnl)
            simulation_results[name]['costs'].append(cost)
            simulation_results[name]['hedge_ratios'].append(new_hedge_ratio)
            simulation_results[name]['entropies'].append(entropy)
            
            current_hedge_ratios[name] = new_hedge_ratio
            
            # Generate the rich proposal with rationale using the LLM
            rich_proposal = get_agent_proposal_with_reasoning(name, new_hedge_ratio, market_view, logger)
            agent_proposals[name] = rich_proposal

        # --- Stage 2: Initiate Debate to get Converged Decision ---
        logger.info(f"--- Initiating Debate for Timestep {i+1} ---")
        converged_hr, cio_rationale, debate_history_for_step = decide_converged_hedge_ratio(
            agent_proposals, market_view, current_hedge_ratios['Agentic DSS'], logger
        )
        logger.info(f"--- Debate Concluded for Timestep {i+1}. Final CIO Rationale: ---")
        logger.info(cio_rationale)
        # Optional: Log or process debate_history_for_step if needed for detailed analysis
        # For example, to just show the last debate round details
        if debate_history_for_step:
            logger.debug("--- Last Debate Round Summary ---")
            last_round = debate_history_for_step[-1]
            logger.debug(f"  CIO Critique: {last_round['cio_feedback']['critique']}")
            for agent_name, proposal in last_round['proposals_after_rebuttal'].items():
                logger.debug(f"  - {agent_name} Final Proposal: {proposal['ratio']:.2f}, Rationale: '{proposal['rationale']}'")
        
        # Calculate PnL and Cost for the Agentic DSS
        pnl_agentic = spot_return - current_hedge_ratios['Agentic DSS'] * futures_return
        cost_agentic = transaction_cost * abs(converged_hr - current_hedge_ratios['Agentic DSS'])
        
        simulation_results['Agentic DSS']['pnls'].append(pnl_agentic)
        simulation_results['Agentic DSS']['costs'].append(cost_agentic)
        simulation_results['Agentic DSS']['hedge_ratios'].append(converged_hr)
        simulation_results['Agentic DSS']['entropies'].append(0) # Entropy not applicable
        simulation_results['Agentic DSS']['rationales'].append(cio_rationale)
        current_hedge_ratios['Agentic DSS'] = converged_hr
        
        # --- Collect Decision Record for History ---
        decision_record = {
            'timestep': i + 1,
            'date': str(market_view['date']) if 'date' in market_view.index else f"Day {i+1}",
            'market_snapshot': {
                # Core prices
                'spot_close': float(market_view['spot_close']),
                'fut_close': float(market_view['fut_close']),
                # Predictions
                'next_day_pred': float(market_view.get('next_day_pred', 0)),
                'next_month_pred': float(market_view.get('next_month_pred', 0)),
                # Volatility metrics
                'spot_std_5': float(market_view.get('spot_std_5', 0)),
                'spot_std_20': float(market_view.get('spot_std_20', 0)),
                'fut_std_20': float(market_view.get('fut_std_20', 0)),
                'vol_regime': int(market_view.get('vol_regime', 1)),
                # Moving averages
                'spot_ma_5': float(market_view.get('spot_ma_5', 0)),
                'spot_ma_20': float(market_view.get('spot_ma_20', 0)),
                'fut_ma_20': float(market_view.get('fut_ma_20', 0)),
                # Basis analysis
                'basis': float(market_view.get('basis', 0)),
                'basis_pct': float(market_view.get('basis_pct', 0)),
                'basis_mean_20': float(market_view.get('basis_mean_20', 0)),
                # Returns
                'spot_ret': float(market_view.get('spot_ret', 0)),
                'fut_ret': float(market_view.get('fut_ret', 0)),
            },
            'agent_proposals': {
                name: {'ratio': float(prop['ratio']), 'rationale': prop['rationale']}
                for name, prop in agent_proposals.items()
            },
            'debate_history': [
                {
                    'round': round_data['round'],
                    'cio_feedback': {
                        'critique': round_data['cio_feedback'].get('critique', ''),
                        'compromise_ratio': float(round_data['cio_feedback'].get('compromise_ratio', 0)),
                        'questions': round_data['cio_feedback'].get('questions_for_agents', {})
                    },
                    'proposals_before': {
                        name: {'ratio': float(p['ratio']), 'rationale': p['rationale']}
                        for name, p in round_data.get('proposals_before_rebuttal', {}).items()
                    },
                    'proposals_after': {
                        name: {'ratio': float(p['ratio']), 'rationale': p['rationale']}
                        for name, p in round_data.get('proposals_after_rebuttal', {}).items()
                    }
                }
                for round_data in (debate_history_for_step or [])
            ],
            'final_decision': {
                'hedge_ratio': float(converged_hr),
                'cio_rationale': cio_rationale
            },
            'performance': {
                'pnl': float(pnl_agentic),
                'cost': float(cost_agentic),
                'net_pnl': float(pnl_agentic - cost_agentic)
            }
        }
        decision_history.append(decision_record)

        # --- Evaluate Baselines ---
        # No Hedge
        simulation_results['No Hedge (Ratio=0.0)']['pnls'].append(spot_return)
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
        
        # --- Save Checkpoint After Each Timestep ---
        checkpoint = {
            'last_completed_timestep': i,
            'simulation_results': simulation_results,
            'current_hedge_ratios': current_hedge_ratios,
            'decision_history': decision_history,
            'timestamp': datetime.now().isoformat()
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, cls=NpEncoder)
        
        # Also save decision_history incrementally for webapp
        with open(decision_history_path, 'w') as f:
            json.dump(decision_history, f, indent=2, cls=NpEncoder)
        
        logger.info(f"💾 Checkpoint saved (timestep {i+1}/{eval_period_len})")

    # --- 4. Calculate and Print Metrics ---
    logger.info("\n--- Final CIO Rationale from last step ---")
    if simulation_results['Agentic DSS']['rationales']:
        logger.info(simulation_results['Agentic DSS']['rationales'][-1])
        
    logger.info("\n--- Comparative Performance Metrics ---")
    for name in all_strategies:
        metrics = calculate_metrics(simulation_results[name])
        logger.info(f"\n--- {name} ---")
        for key, value in metrics.items():
            logger.info(f"{key:<15}: {value: .5f}")

    # --- 5. Plotting ---
    logger.info("\n--- Generating Comparison Plots ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(4, 1, figsize=(18, 32), sharex=True)
    colors = {'Agentic DSS': 'purple', 'Cost Control': 'red', 'Profit Maximization': 'blue', 'Risk Reduction': 'green', 'Static (Ratio=1.0)': 'black', 'No Hedge (Ratio=0.0)': 'orange'}

    # Plot 1: Cumulative Net PnL
    for name in all_strategies:
        net_pnl = np.cumsum(np.array(simulation_results[name]['pnls']) - np.array(simulation_results[name]['costs']))
        axs[0].plot(net_pnl, label=name, color=colors.get(name, 'gray'), linewidth=2 if name == 'Agentic DSS' else 1.5)
    axs[0].set_title('Comparative Net PnL (PnL - Costs)', fontsize=16)
    axs[0].set_ylabel('Cumulative Net PnL', fontsize=12)
    axs[0].legend()
    axs[0].axhline(0, color='black', linestyle='--', linewidth=0.8)

    # Plot 2: Cumulative Costs
    for name in all_strategies:
        axs[1].plot(np.cumsum(simulation_results[name]['costs']), label=name, color=colors.get(name, 'gray'), linewidth=2 if name == 'Agentic DSS' else 1.5)
    axs[1].set_title('Comparative Cumulative Transaction Costs', fontsize=16)
    axs[1].set_ylabel('Cumulative Costs', fontsize=12)
    axs[1].legend()

    # Plot 3: Hedge Ratio Policies
    for name in all_strategies:
        axs[2].plot(simulation_results[name]['hedge_ratios'], label=name, color=colors.get(name, 'gray'), alpha=0.8, drawstyle='steps-post', linewidth=2 if name == 'Agentic DSS' else 1.5)
    axs[2].set_title('Comparative Hedge Ratio Policies', fontsize=16)
    axs[2].set_ylabel('Hedge Ratio', fontsize=12)
    axs[2].legend()

    # Plot 4: Policy Uncertainty (Entropy)
    for name in all_strategies:
        if name not in ['Agentic DSS', 'Static (Ratio=1.0)', 'No Hedge (Ratio=0.0)']:
            axs[3].plot(simulation_results[name]['entropies'], label=name, color=colors.get(name, 'gray'), alpha=0.8)
    axs[3].set_title('Comparative Policy Uncertainty (Entropy)', fontsize=16)
    axs[3].set_ylabel('Entropy', fontsize=12)
    axs[3].set_xlabel('Time Step in Test Set', fontsize=12)
    axs[3].legend()

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_save_path = os.path.join(log_dir, f'simulation_results_{timestamp}.png')
    plt.savefig(final_save_path)
    logger.info(f"\n--- Simulation complete. Plot saved to {final_save_path} ---")
    
    # Final decision history save (already saved incrementally, but ensure final state)
    logger.info(f"--- Decision history saved to {decision_history_path} ({len(decision_history)} records) ---")
    
    # Remove checkpoint file since simulation completed successfully
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("✅ Checkpoint file removed (simulation completed successfully)")
    
    plt.show()

if __name__ == "__main__":
    main()
