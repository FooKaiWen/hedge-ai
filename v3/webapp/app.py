"""
Hedge AI Decision Support System
A Streamlit-based agentic DSS for palm oil hedging with live LLM-mediated agent debates.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import os
from datetime import datetime, timedelta
from google import genai

# === Configuration ===
MODEL = "gemini-2.5-flash"
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'palm_oil_enriched_data.csv')

# === Page Configuration ===
st.set_page_config(
    page_title="Hedge AI | Decision Support System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS for Premium Dark Theme ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        color: #e94560;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #a0a0a0;
        font-size: 1.1rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(233, 69, 96, 0.3);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(233, 69, 96, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e94560, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .agent-card {
        background: linear-gradient(145deg, #1a1f2e, #232838);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid;
        min-height: 280px;
    }
    
    .agent-profit { border-left-color: #22c55e; background: linear-gradient(145deg, #1a2420, #1e2a24); }
    .agent-risk { border-left-color: #eab308; background: linear-gradient(145deg, #242218, #2a2820); }
    .agent-cost { border-left-color: #3b82f6; background: linear-gradient(145deg, #1a1e28, #1e2430); }
    
    .agent-name {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .agent-ratio {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .agent-rationale {
        color: #b0b8c4;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-top: 0.75rem;
    }
    
    .cio-decision {
        background: linear-gradient(135deg, #0f3460, #1a1a2e);
        padding: 2rem;
        border-radius: 16px;
        border: 2px solid #e94560;
        margin-top: 2rem;
    }
    
    .debate-log {
        background: #0d0d15;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Monaco', monospace;
        font-size: 0.85rem;
        max-height: 300px;
        overflow-y: auto;
        color: #a0ffa0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #ff6b6b);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
    }
    
    .sidebar .stNumberInput > div > div > input {
        background: #1a1a2e;
        border: 1px solid #333;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# === LLM Integration ===
@st.cache_resource
def get_llm_client():
    """Initialize the Gemini LLM client."""
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
        if not api_key:
            # Fallback for demo - you should set your own key
            api_key = "AIzaSyBwiRtCXpJZkk2BeP4LPgdHJHGr2hDY404"
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None


def get_agent_proposal(client, agent_name: str, hedge_ratio: float, market_view: dict) -> dict:
    """Generate a rationale for an agent's proposed hedge ratio using LLM with rich market context."""
    prompt = f"""
    You are an expert financial agent with the persona of a '{agent_name}'.
    Your task is to provide a brief, one-sentence rationale for a hedging decision.
    
    Current Market View:
    - CPO Spot Price: {market_view.get('spot_close', 0):.2f} MYR/MT
    - FCPO Futures Price: {market_view.get('fut_close', 0):.2f} MYR/MT
    - FCPO Next Day Prediction: {market_view.get('next_day_pred', 0):.2f} MYR/MT
    - FCPO Next Month Prediction: {market_view.get('next_month_pred', 0):.2f} MYR/MT
    
    Volatility & Risk Metrics:
    - Volatility Regime: {market_view.get('vol_regime', 'Medium')}
    - 20-day Spot Volatility (Std Dev): {market_view.get('spot_std_20', 0):.2f}
    - 5-day Spot Volatility: {market_view.get('spot_std_5', 0):.2f}
    
    Basis Analysis:
    - Basis (Spot - Futures): {market_view.get('basis', 0):.2f}
    - Basis Percentage: {market_view.get('basis_pct', 0):.2f}%
    - 20-day Basis Mean: {market_view.get('basis_mean_20', 0):.2f}
    
    Technical Indicators:
    - 5-day Spot MA: {market_view.get('spot_ma_5', 0):.2f}
    - 20-day Spot MA: {market_view.get('spot_ma_20', 0):.2f}
    - OLS Hedge Ratio (20-day): {market_view.get('hr_ols_20', 0):.4f}
    - Target Hedge Ratio: {market_view.get('target_hr', 0):.2f}
    
    Based on your persona, you have decided on a hedge ratio of: {hedge_ratio:.2f}
    
    Provide a concise, single-sentence justification for this decision, starting with 'As the {agent_name} agent, ...'.
    
    Justification:
    """
    
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        time.sleep(0.5)
        rationale = response.text.strip().replace('\n', ' ')
    except Exception as e:
        rationale = f"As the {agent_name} agent, I propose a hedge ratio of {hedge_ratio:.2f} based on current market conditions."
    
    return {"ratio": hedge_ratio, "rationale": rationale}


def moderate_debate(client, proposals: dict, market_view: dict, current_hedge: float) -> dict:
    """CIO moderates the debate between agents with rich market context."""
    prompt = f"""
    You are a Chief Investment Officer (CIO) moderating a debate between three specialist AI agents.

    Current Market View:
    - CPO Spot Price: {market_view.get('spot_close', 0):.2f} MYR/MT
    - FCPO Futures Price: {market_view.get('fut_close', 0):.2f} MYR/MT
    - FCPO Next Day Prediction: {market_view.get('next_day_pred', 0):.2f} MYR/MT
    - FCPO Next Month Prediction: {market_view.get('next_month_pred', 0):.2f} MYR/MT
    
    Volatility & Risk Metrics:
    - Volatility Regime: {market_view.get('vol_regime', 'Medium')}
    - 20-day Spot Volatility: {market_view.get('spot_std_20', 0):.2f}
    - 5-day Spot Volatility: {market_view.get('spot_std_5', 0):.2f}
    
    Basis Analysis:
    - Basis (Spot - Futures): {market_view.get('basis', 0):.2f}
    - Basis Percentage: {market_view.get('basis_pct', 0):.2f}%
    - 20-day Basis Mean: {market_view.get('basis_mean_20', 0):.2f}
    
    Technical Indicators:
    - 5-day Spot MA: {market_view.get('spot_ma_5', 0):.2f}
    - 20-day Spot MA: {market_view.get('spot_ma_20', 0):.2f}
    - OLS Hedge Ratio (20-day): {market_view.get('hr_ols_20', 0):.4f}
    - Target Hedge Ratio: {market_view.get('target_hr', 0):.2f}
    
    Current Position:
    - Current Hedge Ratio: {current_hedge:.2f}

    Agent Proposals:
    1. Profit Maximization: Ratio={proposals['Profit Maximization']['ratio']:.2f}, Rationale="{proposals['Profit Maximization']['rationale']}"
    2. Risk Reduction: Ratio={proposals['Risk Reduction']['ratio']:.2f}, Rationale="{proposals['Risk Reduction']['rationale']}"
    3. Cost Control: Ratio={proposals['Cost Control']['ratio']:.2f}, Rationale="{proposals['Cost Control']['rationale']}"

    **Your Task**:
    1. Critique: Brief critique of the agents' positions considering the market data
    2. Decision: Decide on a final hedge ratio (0.0 to 1.5)
    3. Rationale: Explain your decision referencing specific market metrics

    **Output Format**: JSON object only
    {{
      "critique": "your critique here",
      "final_ratio": 0.85,
      "cio_rationale": "your rationale here"
    }}
    """
    
    try:
        response = client.models.generate_content(model=MODEL, contents=prompt)
        time.sleep(0.5)
        text = response.text.strip().replace("```json", "").replace("```", "")
        decision = json.loads(text)
        decision['final_ratio'] = np.clip(float(decision.get('final_ratio', 0.5)), 0.0, 1.5)
        return decision
    except Exception as e:
        # Fallback
        avg_ratio = np.mean([p['ratio'] for p in proposals.values()])
        return {
            "critique": "Unable to parse LLM response. Using average of agent proposals.",
            "final_ratio": avg_ratio,
            "cio_rationale": f"Averaging agent proposals due to processing error: {avg_ratio:.2f}"
        }


# === Data Loading ===
@st.cache_data
def load_historical_data():
    """Load historical palm oil data."""
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        st.warning(f"Could not load historical data: {e}")
        return None


# === Main App ===
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Hedge AI</h1>
        <p>Agentic Decision Support System for Palm Oil Hedging</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize LLM
    llm_client = get_llm_client()
    
    # Sidebar - Data Input Form
    with st.sidebar:
        st.markdown("### 📥 Market Data Input")
        st.markdown("---")
        
        # Load data first
        df = load_historical_data()
        
        # Date picker first - this drives the data loading
        if df is not None and len(df) > 0:
            # Get available date range for reference
            df['date_only'] = pd.to_datetime(df['date']).dt.date
            available_dates = sorted(df['date_only'].unique())
            min_data_date = min(available_dates)
            max_data_date = max(available_dates)
            
            input_date = st.date_input(
                "📅 Select Date",
                value=max_data_date,
                help=f"Dataset has data from {min_data_date} to {max_data_date}. Select any date - manual input available if date not found."
            )
            
            # Load data for selected date
            selected_row = df[df['date_only'] == input_date]
            
            if len(selected_row) > 0:
                row = selected_row.iloc[0]
                data_available = True
                
                # Extract all values from the selected date
                default_spot = float(row['spot_close']) if 'spot_close' in row.index else 4200.0
                default_fut = float(row['fut_close']) if 'fut_close' in row.index else 4150.0
                default_next_day = float(row['next_day_pred']) if 'next_day_pred' in row.index else 4180.0
                default_next_month = float(row['next_month_pred']) if 'next_month_pred' in row.index else 4250.0
                spot_std_5 = float(row['spot_std_5']) if 'spot_std_5' in row.index else 50.0
                spot_std_20 = float(row['spot_std_20']) if 'spot_std_20' in row.index else 80.0
                spot_ma_5 = float(row['spot_ma_5']) if 'spot_ma_5' in row.index else 4180.0
                spot_ma_20 = float(row['spot_ma_20']) if 'spot_ma_20' in row.index else 4150.0
                basis = float(row['basis']) if 'basis' in row.index else 50.0
                basis_mean_20 = float(row['basis_mean_20']) if 'basis_mean_20' in row.index else 45.0
                hr_ols_20 = float(row['hr_ols_20']) if 'hr_ols_20' in row.index else 0.85
                target_hr = float(row['target_hr']) if 'target_hr' in row.index else 0.5
                hr_now = float(row['hr_now']) if 'hr_now' in row.index else 0.5
                has_predictions = 'next_day_pred' in row.index and 'next_month_pred' in row.index
                latest_row = row  # For compatibility with rest of code
            else:
                st.warning(f"No data found for {input_date}")
                data_available = False
        else:
            st.error("Dataset not loaded")
            data_available = False
            input_date = datetime.now().date()
        
        # Set defaults if data not available
        if not data_available:
            default_spot, default_fut = 4200.0, 4150.0
            default_next_day, default_next_month = 4180.0, 4250.0
            spot_std_5, spot_std_20 = 50.0, 80.0
            spot_ma_5, spot_ma_20 = 4180.0, 4150.0
            basis, basis_mean_20 = 50.0, 45.0
            hr_ols_20, target_hr, hr_now = 0.85, 0.5, 0.5
            has_predictions = False
            latest_row = None
        
        st.markdown("---")
        
        # Price Data - display as read-only when loaded from dataset
        st.markdown("#### 💹 Price Data")
        col1, col2 = st.columns(2)
        with col1:
            spot_close = st.number_input(
                "CPO Spot (MYR/MT)",
                min_value=0.0,
                max_value=10000.0,
                value=default_spot,
                step=10.0,
                help="CPO spot price",
                key=f"spot_{input_date}"
            )
        with col2:
            fut_close = st.number_input(
                "FCPO Futures (MYR/MT)",
                min_value=0.0,
                max_value=10000.0,
                value=default_fut,
                step=10.0,
                help="FCPO futures price",
                key=f"fut_{input_date}"
            )
        
        # Predictions - auto-derived when data available, manual fallback
        if has_predictions:
            st.markdown("#### 🔮 Predictions (SVR Model)")
            next_day_pred = default_next_day
            next_month_pred = default_next_month
            
            col3, col4 = st.columns(2)
            with col3:
                next_day_change = ((next_day_pred - fut_close) / fut_close * 100) if fut_close > 0 else 0
                day_icon = "📈" if next_day_change >= 0 else "📉"
                st.markdown(f"**Next Day:** {day_icon} RM {next_day_pred:,.0f}")
                st.caption(f"Δ {next_day_change:+.2f}%")
            with col4:
                next_month_change = ((next_month_pred - fut_close) / fut_close * 100) if fut_close > 0 else 0
                month_icon = "📈" if next_month_change >= 0 else "📉"
                st.markdown(f"**Next Month:** {month_icon} RM {next_month_pred:,.0f}")
                st.caption(f"Δ {next_month_change:+.2f}%")
        else:
            st.markdown("#### 🔮 Predictions (Manual)")
            col3, col4 = st.columns(2)
            with col3:
                next_day_pred = st.number_input(
                    "Next Day Pred",
                    min_value=0.0,
                    max_value=10000.0,
                    value=4180.0,
                    step=10.0,
                    help="Predicted FCPO price for tomorrow"
                )
            with col4:
                next_month_pred = st.number_input(
                    "Next Month Pred",
                    min_value=0.0,
                    max_value=10000.0,
                    value=4250.0,
                    step=10.0,
                    help="Predicted FCPO price for next month"
                )
        
        st.markdown("#### 📊 Market Conditions")
        # Auto-derive volatility regime from spot_std_20
        # Thresholds based on historical percentiles: Low (<15), Medium (15-45), High (>=45)
        if spot_std_20 < 15:
            vol_regime = "Low"
            vol_color = "🟢"
        elif spot_std_20 < 45:
            vol_regime = "Medium"
            vol_color = "🟡"
        else:
            vol_regime = "High"
            vol_color = "🔴"
        
        st.markdown(f"**Volatility Regime:** {vol_color} {vol_regime}")
        st.caption(f"Auto-derived from Vol(20d): {spot_std_20:.1f}")
        
        current_hedge = st.slider(
            "Current Hedge Ratio",
            min_value=0.0,
            max_value=1.5,
            value=float(hr_now) if data_available else 0.5,
            step=0.05,
            help="Your current hedge position"
        )
        
        # Calculate derived metrics
        basis_pct = ((spot_close - fut_close) / fut_close * 100) if fut_close > 0 else 0
        
        # Display enriched metrics
        st.markdown("---")
        st.markdown("##### 📈 Enriched Market Data")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"**Basis:** RM {basis:.0f}")
            st.markdown(f"**Basis %:** {basis_pct:.2f}%")
            st.markdown(f"**Vol (5d):** {spot_std_5:.1f}")
        with col_m2:
            st.markdown(f"**MA (5d):** RM {spot_ma_5:.0f}")
            st.markdown(f"**MA (20d):** RM {spot_ma_20:.0f}")
            st.markdown(f"**Vol (20d):** {spot_std_20:.1f}")
        
        st.markdown(f"**OLS Hedge Ratio (20d):** {hr_ols_20:.4f}")
        st.markdown(f"**Previous HR:** {target_hr:.2f}")
        
        st.markdown("---")
        run_analysis = st.button("🚀 Run Agent Debate", use_container_width=True)
    
    # Main Content Area - Charts on top, full width
    # Historical Charts Section
    st.markdown("### 📊 Historical Market Trends")
    
    # Use already loaded df
    if df is not None and len(df) > 0:
        # Get actual data bounds
        data_min_date = df['date'].min().to_pydatetime()
        data_max_date = df['date'].max().to_pydatetime()
        
        # Try to get decision history date range for default
        decision_history_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'decision_history.json')
        default_start = data_min_date
        default_end = data_max_date
        
        if os.path.exists(decision_history_path):
            try:
                with open(decision_history_path, 'r') as f:
                    dh = json.load(f)
                if dh and len(dh) > 0:
                    # Get date range from decision history
                    dh_dates = [pd.to_datetime(d['date']) for d in dh]
                    dh_min = min(dh_dates).to_pydatetime()
                    dh_max = max(dh_dates).to_pydatetime()
                    # Use decision history range, clamped to data bounds
                    default_start = max(data_min_date, dh_min - timedelta(days=7))  # 7 days buffer before
                    default_end = min(data_max_date, dh_max + timedelta(days=7))    # 7 days buffer after
            except:
                pass
        
        # Fallback: if no decision history, use 1 year
        if default_start == data_min_date and default_end == data_max_date:
            default_start = max(data_min_date, data_max_date - timedelta(days=365))
        
        # Date range filter with key based on actual data bounds to prevent stale cache
        st.caption(f"📅 Data available from {data_min_date.strftime('%Y-%m-%d')} to {data_max_date.strftime('%Y-%m-%d')}")
        date_range = st.slider(
            "Select Date Range",
            min_value=data_min_date,
            max_value=data_max_date,
            value=(default_start, default_end),
            format="YYYY-MM-DD",
            key=f"date_slider_{data_min_date.date()}_{data_max_date.date()}_{default_start.date()}"
        )
        
        filtered_df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
        
        # Price Chart - full width
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=("CPO Spot vs FCPO Futures", "Price Predictions"),
            row_heights=[0.65, 0.35]
        )
        
        fig.add_trace(
            go.Scatter(x=filtered_df['date'], y=filtered_df['spot_close'],
                      name="CPO Spot", line=dict(color="#e94560", width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=filtered_df['date'], y=filtered_df['fut_close'],
                      name="FCPO Futures", line=dict(color="#22c55e", width=2)),
            row=1, col=1
        )
        
        if 'next_day_pred' in filtered_df.columns:
            fig.add_trace(
                go.Scatter(x=filtered_df['date'], y=filtered_df['next_day_pred'],
                          name="Next Day Pred", line=dict(color="#3b82f6", width=1.5, dash='dot')),
                row=2, col=1
            )
        
        if 'next_month_pred' in filtered_df.columns:
            fig.add_trace(
                go.Scatter(x=filtered_df['date'], y=filtered_df['next_month_pred'],
                          name="Next Month Pred", line=dict(color="#eab308", width=1.5, dash='dot')),
                row=2, col=1
            )
        
        fig.update_layout(
            height=450,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,26,46,0.8)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=40, b=0),
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Historical data not available. Charts will appear after loading data.")
    
    st.markdown("---")
    
    # Market Overview Cards
    st.markdown("### 📈 Current Market Snapshot")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric("CPO Spot", f"RM {spot_close:,.0f}", delta=None)
    with m2:
        st.metric("FCPO Futures", f"RM {fut_close:,.0f}", delta=f"{basis_pct:+.2f}% basis")
    with m3:
        st.metric("Next Day Pred", f"RM {next_day_pred:,.0f}", 
                 delta=f"{((next_day_pred - fut_close) / fut_close * 100):+.2f}%")
    with m4:
        st.metric("Volatility", vol_regime, delta=None)
    
    st.markdown("---")
    
    # Agent Proposals & Debate
    if run_analysis and llm_client:
        # Build comprehensive market_view with all enriched data
        market_view = {
            # Core prices
            'spot_close': spot_close,
            'fut_close': fut_close,
            'next_day_pred': next_day_pred,
            'next_month_pred': next_month_pred,
            # Volatility metrics
            'vol_regime': vol_regime,
            'spot_std_5': spot_std_5,
            'spot_std_20': spot_std_20,
            # Basis analysis
            'basis': basis,
            'basis_pct': basis_pct,
            'basis_mean_20': basis_mean_20,
            # Technical indicators
            'spot_ma_5': spot_ma_5,
            'spot_ma_20': spot_ma_20,
            # Hedge ratio signals
            'hr_ols_20': hr_ols_20,
            'target_hr': target_hr,
        }
        
        # Agent initial ratios based on personas
        if vol_regime == "High":
            profit_ratio, risk_ratio, cost_ratio = 0.3, 1.2, current_hedge
        elif vol_regime == "Low":
            profit_ratio, risk_ratio, cost_ratio = 0.6, 0.8, current_hedge
        else:
            profit_ratio, risk_ratio, cost_ratio = 0.5, 1.0, current_hedge
        
        with st.spinner("🤖 Agents are deliberating..."):
            # Get agent proposals
            st.markdown("### 🤖 Agent Proposals")
            
            progress = st.progress(0)
            status = st.empty()
            
            status.text("💰 Profit Maximization agent analyzing...")
            profit_proposal = get_agent_proposal(llm_client, "Profit Maximization", profit_ratio, market_view)
            progress.progress(25)
            
            status.text("🛡️ Risk Reduction agent analyzing...")
            risk_proposal = get_agent_proposal(llm_client, "Risk Reduction", risk_ratio, market_view)
            progress.progress(50)
            
            status.text("💵 Cost Control agent analyzing...")
            cost_proposal = get_agent_proposal(llm_client, "Cost Control", cost_ratio, market_view)
            progress.progress(75)
            
            proposals = {
                "Profit Maximization": profit_proposal,
                "Risk Reduction": risk_proposal,
                "Cost Control": cost_proposal
            }
            
            status.text("👔 CIO moderating debate...")
            cio_decision = moderate_debate(llm_client, proposals, market_view, current_hedge)
            progress.progress(100)
            status.empty()
            progress.empty()
        
        # Display Agent Cards
        agent_cols = st.columns(3)
        
        with agent_cols[0]:
            st.markdown(f"""
            <div class="agent-card agent-profit">
                <div class="agent-name" style="color: #22c55e;">💰 PROFIT MAXIMIZATION</div>
                <div class="agent-ratio" style="color: #22c55e;">{profit_proposal['ratio']:.2f}</div>
                <div class="agent-rationale">{profit_proposal['rationale']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with agent_cols[1]:
            st.markdown(f"""
            <div class="agent-card agent-risk">
                <div class="agent-name" style="color: #eab308;">🛡️ RISK REDUCTION</div>
                <div class="agent-ratio" style="color: #eab308;">{risk_proposal['ratio']:.2f}</div>
                <div class="agent-rationale">{risk_proposal['rationale']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with agent_cols[2]:
            st.markdown(f"""
            <div class="agent-card agent-cost">
                <div class="agent-name" style="color: #3b82f6;">💵 COST CONTROL</div>
                <div class="agent-ratio" style="color: #3b82f6;">{cost_proposal['ratio']:.2f}</div>
                <div class="agent-rationale">{cost_proposal['rationale']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # CIO Final Decision
        st.markdown("---")
        st.markdown("### 👔 CIO Final Decision")
        
        decision_col1, decision_col2 = st.columns([1, 2])
        
        with decision_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{cio_decision['final_ratio']:.2f}</div>
                <div class="metric-label">Recommended Hedge Ratio</div>
            </div>
            """, unsafe_allow_html=True)
        
        with decision_col2:
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #1a1f2e, #232838); padding: 1.25rem; border-radius: 12px; margin-bottom: 0.75rem; border-left: 4px solid #eab308;">
                <strong style="color: #eab308;">CIO Critique:</strong>
                <p style="color: #b0b8c4; margin: 0.5rem 0 0 0; line-height: 1.6;">{cio_decision.get('critique', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #1a2420, #1e2a24); padding: 1.25rem; border-radius: 12px; border-left: 4px solid #22c55e;">
                <strong style="color: #22c55e;">Decision Rationale:</strong>
                <p style="color: #b0b8c4; margin: 0.5rem 0 0 0; line-height: 1.6;">{cio_decision.get('cio_rationale', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Store in session state for chart
        st.session_state['last_recommendation'] = {
            'date': input_date,
            'ratio': cio_decision['final_ratio'],
            'spot': spot_close,
            'futures': fut_close
        }
    
    elif not run_analysis:
        st.info("👈 Fill in the market data and click **Run Agent Debate** to get a hedge ratio recommendation.")
    
    # === Decision History Section ===
    st.markdown("---")
    st.markdown("### 📜 Decision History")
    st.markdown("*Historical LLM-mediated hedge decisions from simulation runs*")
    
    # Load decision history
    decision_history_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'decision_history.json')
    
    if os.path.exists(decision_history_path):
        with open(decision_history_path, 'r') as f:
            decision_history = json.load(f)
        
        if decision_history:
            st.markdown(f"**{len(decision_history)} decisions loaded from simulation**")
            
            # Timeline chart
            history_df = pd.DataFrame([
                {
                    'Date': d['date'][:10] if len(d['date']) > 10 else d['date'],
                    'Hedge Ratio': d['final_decision']['hedge_ratio'],
                    'Net PnL': d['performance']['net_pnl'],
                    'Vol Regime': ['Low', 'Medium', 'High'][d['market_snapshot']['vol_regime']]
                }
                for d in decision_history
            ])
            
            # Hedge Ratio Timeline Chart
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=history_df['Date'],
                y=history_df['Hedge Ratio'],
                mode='lines+markers',
                name='CIO Hedge Ratio',
                line=dict(color='#e94560', width=2),
                marker=dict(size=6)
            ))
            fig_timeline.update_layout(
                title='Historical CIO Hedge Ratio Decisions',
                xaxis_title='Date',
                yaxis_title='Hedge Ratio',
                height=300,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                font=dict(family="Inter, sans-serif")
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Cumulative PnL Chart
            history_df['Cumulative PnL'] = history_df['Net PnL'].cumsum()
            history_df['Cumulative PnL (%)'] = history_df['Cumulative PnL'] * 100
            
            fig_pnl = go.Figure()
            
            # Cumulative PnL line
            fig_pnl.add_trace(go.Scatter(
                x=history_df['Date'],
                y=history_df['Cumulative PnL (%)'],
                mode='lines+markers',
                name='Agentic DSS Cumulative PnL',
                line=dict(color='#22c55e', width=2),
                marker=dict(size=5),
                fill='tozeroy',
                fillcolor='rgba(34, 197, 94, 0.1)'
            ))
            
            # Add zero line
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="#666", opacity=0.5)
            
            # PnL per decision (bar overlay)
            colors = ['#22c55e' if pnl >= 0 else '#ef4444' for pnl in history_df['Net PnL']]
            fig_pnl.add_trace(go.Bar(
                x=history_df['Date'],
                y=history_df['Net PnL'] * 100,
                name='Per-Decision PnL',
                marker_color=colors,
                opacity=0.4,
                yaxis='y2'
            ))
            
            fig_pnl.update_layout(
                title='📈 LLM Hedge Strategy Performance (Cumulative PnL)',
                xaxis_title='Date',
                yaxis_title='Cumulative PnL (%)',
                yaxis2=dict(
                    title='Per-Decision PnL (%)',
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                height=350,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                font=dict(family="Inter, sans-serif"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified'
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
            
            # Summary metrics
            total_pnl = history_df['Net PnL'].sum() * 100
            avg_pnl = history_df['Net PnL'].mean() * 100
            win_rate = (history_df['Net PnL'] > 0).sum() / len(history_df) * 100
            max_drawdown = (history_df['Cumulative PnL'].cummax() - history_df['Cumulative PnL']).max() * 100
            
            pnl_cols = st.columns(4)
            with pnl_cols[0]:
                st.metric("Total PnL", f"{total_pnl:+.2f}%", delta_color="normal")
            with pnl_cols[1]:
                st.metric("Avg PnL/Decision", f"{avg_pnl:+.4f}%")
            with pnl_cols[2]:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with pnl_cols[3]:
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse")
            
            st.markdown("---")
            
            # Decision Cards - show recent decisions
            st.markdown("#### Recent Decisions")
            
            # Pagination
            decisions_per_page = 15
            total_pages = max(1, (len(decision_history) + decisions_per_page - 1) // decisions_per_page)
            page = st.selectbox("Page", range(1, total_pages + 1), index=total_pages - 1, 
                               format_func=lambda x: f"Page {x} of {total_pages}")
            
            start_idx = (page - 1) * decisions_per_page
            end_idx = min(start_idx + decisions_per_page, len(decision_history))
            
            # Show decisions in reverse order (most recent first within page)
            for idx in range(end_idx - 1, start_idx - 1, -1):
                d = decision_history[idx]
                vol_label = ['Low', 'Medium', 'High'][d['market_snapshot']['vol_regime']]
                
                with st.expander(f"📅 **{d['date'][:10] if len(d['date']) > 10 else d['date']}** | Hedge Ratio: **{d['final_decision']['hedge_ratio']:.2f}** | Net PnL: {d['performance']['net_pnl']:.5f}"):
                    
                    # Market Snapshot
                    st.markdown("**Market Conditions:**")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        st.metric("Spot", f"RM {d['market_snapshot']['spot_close']:,.0f}")
                    with mc2:
                        st.metric("Futures", f"RM {d['market_snapshot']['fut_close']:,.0f}")
                    with mc3:
                        st.metric("Volatility", vol_label)
                    with mc4:
                        st.metric("Basis %", f"{d['market_snapshot']['basis_pct']:.2f}%")
                    
                    st.markdown("---")
                    
                    # Agent Proposals
                    st.markdown("**Agent Proposals:**")
                    for agent_name, proposal in d['agent_proposals'].items():
                        color = {'Profit Maximization': '#22c55e', 'Risk Reduction': '#eab308', 'Cost Control': '#3b82f6'}.get(agent_name, '#888')
                        st.markdown(f"""
                        <div style="background: linear-gradient(145deg, #1a1f2e, #232838); padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {color};">
                            <strong style="color: {color};">{agent_name}</strong> <span style="color: #ffffff;">→ Ratio: <strong style="color: #ffffff; font-size: 1.1em;">{proposal['ratio']:.2f}</strong></span><br>
                            <span style="color: #b0b8c4; font-size: 0.9em;">{proposal['rationale']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Debate History (if available)
                    if d.get('debate_history') and len(d['debate_history']) > 0:
                        st.markdown("---")
                        st.markdown(f"**Debate Rounds:** {len(d['debate_history'])} round(s)")
                        for round_data in d['debate_history']:
                            with st.expander(f"Round {round_data['round']}", expanded=False):
                                # CIO Feedback
                                if round_data.get('cio_feedback'):
                                    st.markdown("**CIO Critique:**")
                                    st.markdown(f"> {round_data['cio_feedback'].get('critique', 'N/A')}")
                                    st.markdown(f"*Compromise Ratio:* **{round_data['cio_feedback'].get('compromise_ratio', 'N/A')}**")
                                
                                # Questions
                                if round_data.get('cio_feedback', {}).get('questions'):
                                    st.markdown("**CIO Questions:**")
                                    for agent, question in round_data['cio_feedback']['questions'].items():
                                        st.markdown(f"- *{agent}:* {question[:200]}..." if len(question) > 200 else f"- *{agent}:* {question}")
                                
                                # Updated Proposals
                                if round_data.get('proposals_after'):
                                    st.markdown("**Updated Proposals:**")
                                    for agent_name, proposal in round_data['proposals_after'].items():
                                        st.markdown(f"- **{agent_name}:** {proposal['ratio']:.2f}")
                    
                    # CIO Decision
                    st.markdown("---")
                    st.markdown("**CIO Decision:**")
                    if d.get('debate_summary'):
                        st.markdown(f"*Critique:* {d['debate_summary']}")
                    st.markdown(f"**Rationale:** {d['final_decision']['cio_rationale']}")
        else:
            st.info("No decision history found. Run the simulation to generate history.")
    else:
        st.info("📊 No decision history available. Run `python v3/main_agentic_dss.py` to generate historical decisions.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.85rem;">
        <p>Hedge AI Decision Support System | Powered by Gemini LLM | Palm Oil Hedging Analytics</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
