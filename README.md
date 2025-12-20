
# 📉 Enterprise Bond Market Sentiment Analysis System
### *Predicting Liquidity Crises using FINRA API Data & Ensemble ML*

**Author:** divishathred-ops
**Last Updated:** 2025-12-20

---

## 🚀 Executive Summary
This project implements an **enterprise-grade Machine Learning pipeline** designed to detect early warning signs of liquidity crises in the OTC (Over-The-Counter) Bond Market.

Unlike the stock market, the bond market is opaque and illiquid. By connecting directly to the **FINRA Fixed Income API**, this system ingests raw transaction data, engineers advanced financial features, and compares two distinct AI architectures (**LSTM** vs. **XGBoost**) to classify market stress levels in real-time.

**Key Achievement:** Successfully identified "Crisis" events with high precision by analyzing Breadth Momentum and Sentiment Ratios.

---

## 🧠 The Core Logic: How It Works

### 1. Data Cleaning Pipeline (`DataCleaner` Class)
Raw financial data is notoriously "dirty." The system applies a rigorous cleaning protocol before any ML takes place:
* **Forward-Fill Imputation:** Bond trades are sporadic. If no price exists for a timestamp, the system assumes the last traded price is valid (Market Memory logic).
* **IQR Outlier Removal:** "Fat finger" errors or reporting glitches are removed using the Interquartile Range method to prevent skewing the model.
* **Dimensionality Reduction:** Columns with >60% missing data are automatically dropped to reduce noise.

### 2. Feature Engineering (The "Secret Sauce")
We transform raw rows into 22 predictive signals. The models don't look at prices; they look at **market psychology**.

| Feature Category | Key Feature | The Logic / Formula | Why It Matters |
| :--- | :--- | :--- | :--- |
| **Market Health** | `Breadth` | `(Advances - Declines) / Total Issues` | Determines if the *majority* of the market is strong, or just a few large companies. |
| **Investor Psychology** | `Buy Ratio` | `Buy Vol / (Buy Vol + Sell Vol)` | **0.2** = Panic Selling. **0.8** = Greed. This acts as a "Fear Gauge." |
| **Velocity** | `Sentiment Shift` | `Current Buy Ratio - Yesterday's Buy Ratio` | Detects *how fast* investors are changing their minds. Rapid drops signal crashes. |
| **Volume Stress** | `Volume Ratio` | `Current Vol / 20-Day Moving Avg` | Low volume ratios often precede liquidity freezes (crises). |

### 3. Model Architecture Comparison
We trained two competing models to see which logic applies best to bond markets.

#### 🏛️ Model A: LSTM (Long Short-Term Memory)
* **The Approach:** Deep Learning (Neural Network).
* **The Strategy:** Looks at the last **15 days** as a "movie" sequence.
* **Strengths:** Good at catching trends that build up slowly over time.
* **Weakness:** Can struggle with "Flash Crashes" that have no historical buildup.

#### 🌳 Model B: XGBoost (Gradient Boosted Trees)
* **The Approach:** Ensemble Decision Trees.
* **The Strategy:** "20 Questions" Logic. It splits data based on immediate thresholds.
* **The Decision Pathways (3 Key Scenarios):**
    Instead of a black box, we can trace the exact "IF-THEN" logic the model uses to flag a crisis:

    **Logic 1: The Liquidity Freeze (High Severity)**
    * `IF Buy_Ratio < 0.35` (Panic Selling Dominates)
    * `AND Volume_Ratio < 0.60` (Market Liquidity Dries Up)
    * `THEN -> PREDICT CRISIS (Class 2)`
    * *Insight:* When sellers overwhelm buyers AND trading volume vanishes, prices gap down instantly. This is the most dangerous state for a bond trader.

    **Logic 2: Broad Sector Collapse (Systemic Risk)**
    * `IF Breadth < -0.50` (More decliners than advancers)
    * `AND Breadth_Momentum < -0.1` (The selling is accelerating)
    * `THEN -> PREDICT WARNING (Class 1)`
    * *Insight:* A negative breadth trend shows the weakness is systemic, not just isolated to one bad bond. This catches "contagion" events.

    **Logic 3: The Volatility Trap (Uncertainty)**
    * `IF Sentiment_Vol > 2.0` (Extreme swings in buyer confidence)
    * `AND Total_Vol > 1.5` (Massive churning of positions)
    * `THEN -> PREDICT WARNING (Class 1)`
    * *Insight:* High volume with high volatility usually indicates a market "top" or a pivot point before a crash.

* **Strengths:** **Winner for this project.** It handles "Event-Driven" data better. It identifies specific *combinations* of features that trigger a crisis immediately, without needing a long history.

---

## 📊 Visualizations & Results

The pipeline generates interactive HTML dashboards (Plotly) to visualize performance.

1.  **Confusion Matrix:** Shows exactly where the model confused "Warning" signals with "Normal" days.
2.  **Training History:** Plots the Neural Network's learning curve over 100 epochs.

### 📥 How to View Results
The visualization files are located in the `Visualizations/` folder of this repository.
* `lstm_confusion_matrix.html`
* `model_comparison.html`

**Download these files and open them in any web browser to interact with the data.**

---

## 💻 Code Structure

* **`STEP 0-3`**: **Authentication & Extraction.** Connects to FINRA API using OAuth2 to fetch `CorporateBreadth` and `TreasuryAggregates`.
* **`STEP 4`**: **Cleaning.** Runs the `DataCleaner` class to sanitize 50,000+ records.
* **`STEP 5-6`**: **Engineering.** Calculates the 22 features and generates the `TARGET` variable (0=Normal, 1=Warning, 2=Crisis).
* **`STEP 7-11`**: **Training.** Splits data (60/20/20), balances class weights, and trains LSTM/XGBoost.
* **`STEP 12-14`**: **Deployment.** Saves the best model (`.h5` or `.json`) and pushes results to GitHub.

---

## 🔮 Conclusion
The **XGBoost** model proved superior in detecting liquidity crises due to its ability to handle "regime changes" (sudden market shifts) by analyzing the *current state* of Buy Ratios and Market Breadth, whereas the LSTM struggled to find consistent sequential patterns in the chaotic OTC data.

