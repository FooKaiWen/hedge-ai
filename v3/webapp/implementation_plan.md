# Hedge AI Decision Support System - Streamlit App

## Overview
A Streamlit-based dashboard for palm oil hedging that displays LLM-mediated hedge ratio recommendations and historical market trends.

## Proposed Changes

### [NEW] `v3/webapp/app.py`
Single Streamlit app file containing:

**Page Layout:**
1. **Hero Metrics** - Large display of recommended hedge ratio + key market indicators
2. **Agent Debate Panel** - Cards showing each agent's proposal and rationale
3. **Time Series Charts** - Interactive Plotly charts for:
   - CPO Spot vs FCPO Futures prices
   - Historical hedge ratios
   - Price predictions
4. **Data Table** - Filterable view of raw market data

**Features:**
- Load data from existing `palm_oil_data_cleaned.csv`
- Mock/simulate agent recommendations (no live LLM calls for demo)
- Date range selector for time series
- Responsive premium dark theme

---

## Verification Plan
```bash
cd v3/webapp && streamlit run app.py
```
