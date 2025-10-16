import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class HedgeEnv(gym.Env):
    """
    Custom Gym environment for the Palm Oil Hedging problem.
    Supports three agent reward strategies: profit, sharpe, and cost.
    Automatically tracks performance metrics (ROI, Sharpe, Cost Efficiency).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, forecast_nextday_model, forecast_nextmonth_model,
                 start_date, end_date,
                 initial_cash=1_000_000,
                 lot_size=25,
                 transaction_cost_pct=0.001,
                 max_episode_steps=90,
                 risk_aversion=0.01,
                 reward_strategy='profit'):
        super(HedgeEnv, self).__init__()

        self.data = data.reset_index(drop=True)
        self.forecast_nextday_model = forecast_nextday_model
        self.forecast_nextmonth_model = forecast_nextmonth_model
        self.start_date = pd.to_datetime(start_date, dayfirst=True).timestamp()
        self.end_date = pd.to_datetime(end_date, dayfirst=True).timestamp()
        self.initial_cash = initial_cash
        self.lot_size = lot_size
        self.transaction_cost_pct = transaction_cost_pct
        self.risk_aversion = risk_aversion
        self.reward_strategy = reward_strategy

        self.simulation_data = self.data[
            (self.data['datetime'] >= self.start_date) &
            (self.data['datetime'] <= self.end_date)
        ].reset_index(drop=True)

        self.max_episode_steps = max_episode_steps

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, 0, 0, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.start_index = self.np_random.integers(
            low=0,
            high=len(self.simulation_data) - self.max_episode_steps
        )
        self.current_step = self.start_index
        self.episode_step_count = 0

        # Portfolio state
        self.cash = self.initial_cash
        self.cpo_inventory = 0
        self.futures_positions = 0
        self.portfolio_value = self.initial_cash

        # Tracking metrics
        self.daily_returns = []
        self.portfolio_values = [self.portfolio_value]  # ✅ newly added
        self.transaction_costs = []  # ✅ newly added
        self.history = []

        return self._get_observation(), self._get_info()

    def _get_observation(self):
        if self.current_step >= len(self.simulation_data):
            return np.zeros(self.observation_space.shape)

        current_price = self.simulation_data.iloc[self.current_step]['close']
        current_data = self.simulation_data.iloc[self.current_step].to_frame().T

        try:
            forecast_nextday_price = self.forecast_nextday_model.predict(current_data)[0]
            forecast_nextmonth_price = self.forecast_nextmonth_model.predict(current_data)[0]
        except Exception:
            forecast_nextday_price = current_price
            forecast_nextmonth_price = current_price

        obs = np.array([
            current_price,
            forecast_nextday_price,
            forecast_nextmonth_price,
            self.cash,
            self.cpo_inventory,
            self.futures_positions
        ], dtype=np.float32)
        return obs

    def _get_info(self):
        return {
            "current_step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "futures_positions": self.futures_positions
        }

    def step(self, action):
        hedge_ratio = float(np.clip(action[0], 0.0, 1.0))
        current_price = self.simulation_data.iloc[self.current_step]['close']
        next_price = self.simulation_data.iloc[
            min(self.current_step + 1, len(self.simulation_data) - 1)
        ]['close']

        forecast_horizon = 90
        forecast_daily = 100
        expected_production = forecast_daily * forecast_horizon
        target_futures_contracts = (hedge_ratio * expected_production) / self.lot_size
        contracts_to_trade = target_futures_contracts - self.futures_positions

        transaction_cost = abs(contracts_to_trade) * self.lot_size * current_price * self.transaction_cost_pct
        self.cash -= transaction_cost
        self.futures_positions = target_futures_contracts

        # Margin requirement
        margin_per_contract = 8000
        margin_required = self.futures_positions * margin_per_contract
        margin_penalty = 0.0
        if margin_required > self.cash:
            shortfall = margin_required - self.cash
            margin_penalty = shortfall * 0.1
            self.cash -= margin_penalty

        # Daily sales and futures PnL
        daily_cpo_production = self.np_random.uniform(80, 120)
        daily_sales = daily_cpo_production * current_price
        self.cash += daily_sales

        futures_pnl = (current_price - next_price) * self.futures_positions * self.lot_size
        self.cash += futures_pnl

        hedging_cost = self.futures_positions * self.lot_size * current_price * 0.00001
        self.cash -= hedging_cost

        prev_value = self.portfolio_value
        self.portfolio_value = self.cash + (self.cpo_inventory * current_price)
        step_pnl = self.portfolio_value - prev_value
        step_return = step_pnl / prev_value if prev_value > 0 else 0.0

        self.daily_returns.append(step_return)
        self.portfolio_values.append(self.portfolio_value)  # ✅ track continuously
        self.transaction_costs.append(transaction_cost + hedging_cost + margin_penalty)

        # Reward logic
        if self.reward_strategy == 'profit':
            reward = step_pnl
        elif self.reward_strategy == 'sharpe':
            reward = step_pnl - self.risk_aversion * (step_pnl ** 2)
        elif self.reward_strategy == 'cost':
            reward = -(transaction_cost + hedging_cost + margin_penalty)
        else:
            raise ValueError(f"Unknown reward strategy: {self.reward_strategy}")

        self.current_step += 1
        self.episode_step_count += 1
        terminated = self.current_step >= len(self.simulation_data) - 1
        truncated = self.episode_step_count >= self.max_episode_steps

        obs = self._get_observation()
        info = self._get_info()
        info.update({
            "step_pnl": step_pnl,
            "step_return": step_return,
            "hedge_ratio": hedge_ratio,
            "margin_required": margin_required,
            "margin_penalty": margin_penalty
        })

        return obs, reward, terminated, truncated, info

    def compute_metrics(self):
        if len(self.portfolio_values) < 2:
            return {"ROI": 0.0, "Sharpe Ratio": 0.0, "Cost Efficiency Ratio": 0.0}

        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        roi = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        total_cost = np.sum(self.transaction_costs)
        cost_efficiency = ((portfolio_values[-1] - portfolio_values[0]) / total_cost) if total_cost > 0 else 0.0

        return {
            "ROI": float(np.nan_to_num(roi)),
            "Sharpe Ratio": float(np.nan_to_num(sharpe_ratio)),
            "Cost Efficiency Ratio": float(np.nan_to_num(cost_efficiency))
        }

    def render(self, mode='human', close=False):
        if close:
            return
        print(f"Step {self.current_step} | Value: {self.portfolio_value:,.2f} | Cash: {self.cash:,.2f}")

    def close(self):
        pass
