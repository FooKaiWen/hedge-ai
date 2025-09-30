import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class HedgeEnv(gym.Env):
    """
    Custom Gym environment for the Palm Oil Hedging problem.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data, forecast_nextday_model, forecast_nextmonth_model,
                 start_date, end_date,
                 initial_cash=1_000_000,
                 lot_size=25,  # Metric tons per futures contract
                 transaction_cost_pct=0.001,
                 max_episode_steps=90,
                 risk_aversion=0.01
                 ):
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

        # Filter data for the simulation period
        self.simulation_data = self.data[
            (self.data['datetime'] >= self.start_date) & 
            (self.data['datetime'] <= self.end_date)
        ].reset_index(drop=True)

        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.episode_step_count = 0

        # Define the state space (observation space)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, 0, 0, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )

        # Define the action space (hedge ratio between 0 and 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick a random start index so episodes are not always identical
        self.start_index = self.np_random.integers(
            low=0, 
            high=len(self.simulation_data) - self.max_episode_steps
        )
        self.current_step = self.start_index
        self.episode_step_count = 0

        # Reset portfolio
        self.cash = self.initial_cash
        self.cpo_inventory = 0
        self.futures_positions = 0
        self.portfolio_value = self.initial_cash
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
            "cpo_inventory": self.cpo_inventory,
            "futures_positions": self.futures_positions
        }

    def step(self, action):
        hedge_ratio = float(np.clip(action[0], 0.0, 1.0))

        # Current and next day spot prices
        current_price = self.simulation_data.iloc[self.current_step]['close']
        next_price = self.simulation_data.iloc[
            min(self.current_step + 1, len(self.simulation_data) - 1)
        ]['close']

        # --- 1. Forecast 3-month production (90 days) ---
        forecast_horizon = 90
        forecast_daily = 100  # base forecast (tonnes/day), can be dynamic
        expected_production = forecast_daily * forecast_horizon

        # --- 2. Hedge expected production with FCPO contracts ---
        target_futures_contracts = (hedge_ratio * expected_production) / self.lot_size
        contracts_to_trade = target_futures_contracts - self.futures_positions

        transaction_cost = abs(contracts_to_trade) * self.lot_size * current_price * self.transaction_cost_pct
        self.cash -= transaction_cost
        self.futures_positions = target_futures_contracts

        # --- 3. Daily production realization (stochastic) ---
        daily_cpo_production = self.np_random.uniform(80, 120)

        # Spot sales: all production sold same day
        daily_sales = daily_cpo_production * current_price
        self.cash += daily_sales

        # --- 4. Futures mark-to-market PnL ---
        # (in reality only realized at expiry, but FCPO is margin-settled daily)
        futures_pnl = (current_price - next_price) * self.futures_positions * self.lot_size
        self.cash += futures_pnl

        # --- 5. Hedging cost (margin/holding penalty) ---
        hedging_cost = self.futures_positions * self.lot_size * current_price * 0.00001
        self.cash -= hedging_cost

        # --- 6. Portfolio update ---
        prev_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash  # no big inventories anymore

        step_pnl = self.portfolio_value - prev_portfolio_value

        # --- 7. Reward: portfolio growth (could change to variance-reduction metric) ---
        reward = step_pnl

        # Advance step
        self.current_step += 1
        self.episode_step_count += 1

        # Termination vs truncation
        terminated = self.current_step >= len(self.simulation_data) - 1
        truncated = self.episode_step_count >= self.max_episode_steps

        if terminated:
            print("--- Episode Terminated ---")
        if truncated:
            print("--- Episode Truncated (90-day limit reached) ---")

        obs = self._get_observation()
        info = self._get_info()
        info.update({
            "step_pnl": step_pnl,
            "reward_unscaled": step_pnl,
            "hedge_ratio": hedge_ratio,
            "futures_positions": self.futures_positions
        })

        return obs, reward, terminated, truncated, info

    def render(self, mode='human', close=False):
        if close:
            return
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: {self.portfolio_value:,.2f}")
        print(f"Cash: {self.cash:,.2f}")
        print(f"CPO Inventory: {self.cpo_inventory:,.2f} MT")
        print(f"Futures Positions: {self.futures_positions} contracts")
        print("-" * 30)

    def close(self):
        pass
