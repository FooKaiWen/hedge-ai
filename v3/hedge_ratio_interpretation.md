# Hedge Ratio Interpretation in CPO Hedging

This document clarifies the interpretation of hedge ratios, particularly focusing on their meaning as a percentage of inventory, and the implications of negative or greater-than-one values.

### 1. Hedge Ratio as a Percentage of Current Inventory

**Yes, this is the most practical and common way to interpret the hedge ratio.**

If you have a certain quantity of Crude Palm Oil (CPO) in your inventory, a hedge ratio tells you what proportion of that physical holding you are trying to cover in the futures market.

*   **Example:** If you own 1,000 metric tons of CPO and the agent recommends a hedge ratio of **0.8**:
    *   This means you intend to hedge **80%** of your current physical inventory.
    *   **Action:** You would go to the futures market and **sell** futures contracts equivalent to 800 metric tons of CPO.
    *   **Purpose:** This establishes an offsetting position. If the price of CPO drops, the loss in value of your physical inventory is partially offset by the profit from your short futures position.

### 2. Interpretation of a Negative Hedge Ratio (e.g., -0.5)

A negative hedge ratio signifies a move from hedging into pure speculation.

*   **Action:** With a negative hedge ratio (e.g., -0.5), you would **buy** futures contracts equivalent to 50% of your physical inventory.
*   **Financial Position:** You are now long on your physical CPO *and* long on futures contracts. Instead of offsetting risk, you are **amplifying your exposure** to price movements.
*   **Intent:** This is a **speculative bet that the price of CPO will go up significantly.**
    *   If the price rises, you profit from both your physical CPO and your long futures position, leading to magnified gains.
    *   If the price falls, you lose money on both, leading to magnified losses.
*   **Conclusion:** A negative hedge ratio is a purely **speculative action** that increases risk in pursuit of higher potential profit, rather than a hedging action to reduce risk.

### 3. Interpretation of a Hedge Ratio Greater Than 1.0 (e.g., 1.5)

A hedge ratio greater than 1.0 is referred to as **over-hedging**.

*   **Action:** If you own 1,000 metric tons of CPO and the agent recommends a hedge ratio of **1.5**:
    *   This means you intend to hedge **150%** of your current physical inventory.
    *   **Action:** You would **sell** futures contracts equivalent to 1,500 metric tons of CPO.
*   **Financial Position:**
    *   The first 1,000 tons of short futures act as a perfect hedge for your 1,000 tons of physical CPO.
    *   The remaining 500 tons of short futures (1,500 - 1,000) constitute a **naked short position**.
*   **Intent:** This is a **speculative bet that the price of CPO will go down significantly.**
    *   If the price falls, your loss on the physical inventory is more than offset by gains from your larger short futures position, resulting in a net profit.
    *   If the price rises, your gain on the physical inventory is more than negated by the losses on your larger short futures position, resulting in a net loss.
*   **Conclusion:** Over-hedging is a **speculative action** designed to profit from an anticipated price *decline*. While it still involves selling futures (like a hedge), the quantity exceeds the physical exposure, turning the excess into a directional bet.

In summary, while a hedge ratio of 0 to 1.0 is generally considered for risk management, values outside this range (like negative values or values above 1.0) inherently involve taking on **speculative positions** to either profit from anticipated price increases (negative ratio) or decreases (over-hedging).