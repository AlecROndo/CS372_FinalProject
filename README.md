# Bitcoin Derivatives Trading System

## What it Does

This project builds a machine learning pipeline that predicts the price of Bitcoin binary options on the [Kalshi](https://kalshi.com) prediction market. It collects live candlestick data from the Kalshi API and 1-minute BTC price data from Coinbase, engineers features like rolling volatility windows, strike distance, and time-to-expiry, then trains and evaluates five models — Lasso (Linear), Lasso (Poly-3), XGBoost, LightGBM, and a 4-layer NN + LSTM — to forecast market prices. A backtesting engine uses those predictions to simulate long/short trading strategies across historical events, starting with $1,000 of capital and sizing positions dynamically based on model confidence.

---
## Design Choices

The project evolved in two distinct phases.

**Phase 1 — Options-style price prediction:** The original goal was to predict the Kalshi market price using only current BTC price data and time remaining until expiry, similar to how options pricing models (e.g. Black-Scholes) derive contract value from the underlying asset price, strike distance, and time-to-expiry. Custom features like rolling volatility windows, strike distance, and a time-decay term were engineered to capture this relationship as detailed as possible. The sklearn models (Linear Regression, Poly, XGBoost, LightGBM) were the primary models of this phase and served as a baseline for how well contract price could be explained by current market metrics without any forecasting.

Details:
Linear Regression's main strength is its interpretability. There are simple coefficients that tell the exact way they contribute to the model estimation. This allows for high level of understanding and interpretability of what the model does and how it calculuates it. Lasso further improves this by ignoring and essentialyl deleting coefficients that are negligable or noise. 

Polynomial Regression:
This attempts to furhter improve upon Linear regression by capturing quadratic and nonlinear relations between the prediction and the features. This allows for more complex relationships between numerical features.

XGBoost: 
Next, I attempted to use XGBoost. After research, I determined that XGBoost was strong at finding nonlinear feature relations automatically without needing to directly establish them myself. This allows the model to find more complex relationships. It also has regularization built in, and is overall, very capable model when it comes to complex relationship such as options/contract pricing.

LightGBM:
Lastly, I researched LightGBM, and found that it is essentially an improved version of XGBoost. It has speed, memory improvements, and often trains faster, and finds more accurate relationships between the two. The results did show that they had fundamentally very different scores and well within the margin of error/variance.

**Phase 2 — Sequential prediction with NN/LSTM:** After establishing that the options-style features could meaningfully explain price, the question became whether it was possible to actually profit from predictions. Basically, whether future price movements were predictable from historical sequences of those same features. I decided to add a NN/LSTM model that takes a sliding window of past observations as input and tries to forecast where the market price is headed, enabling a long/short strategy based off the predictions that my model produced, allowing for fully systemically trading without any human supervision.

My decision to use LSTM for the second part of this project was because of its temporal understanding and sliding window. It takes previous data and to predict future outcomes. It can detect lagged returns, find time series/variable length related patterns, and can add time decay, where it essentually discounts past information more heavily than recent info. These are all relationships that tradiitonal models with Sklearn could not analyze. 

That being said, because it does incorporate a NN, it is extremely data hungry, and required me to import and API call substantially more markets than with the traditional models.
---


## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect market and BTC data
python models_data/dataMarketCollect.py

# 3. Evaluate sklearn models (cross-validated RMSE)
python models_data/sklearn_test.py

# 4. Run sklearn backtest
python models_data/backtester.py

# 5. Analyze and plot results
python models_data/DataAnalyze.py

# 6. LSTM model — open in Google Colab
#    models_data/FinalProject (8).ipynb
```

See SETUP.md for full setup instructions including API credential configuration.

---

## Video Links

**Demo:**
https://drive.google.com/file/d/17-nkELp7PENR0TLnsPZc_msAcIOx0pdp/view?usp=sharing

**Technical Walkthrough:**
https://drive.google.com/file/d/1FOsoHb5oA_MX3nSg86j8KBV8c6jWPSj-/view?usp=sharing

---

## Evaluation

### sklearn Models — 7-fold cross-validated RMSE (19 events)

| Model | Mean Test RMSE |
|---|---|
| Lasso (Linear) | 0.1929 |
| Lasso (Poly-3) | 0.1446 |
| XGBoost | 0.0810 |
| LightGBM | 0.0797 |

RMSE is on a 0–1 scale (market prices are probabilities). LightGBM and XGBoost are roughly 2.4× better than the linear baseline. Chart saved to [rmse_by_model.png](rmse_by_model.png).

### LSTM Model — Train/Val/Test split (FinalProject notebook)

| Split | MAE | RMSE |
|---|---|---|
| Train | 0.1367 | 0.1709 |
| Val | 0.1373 | 0.1702 |
| Test | 0.1425 | 0.1721 |
| Naive baseline (test) | 0.0744 | 0.1303 |

4-layer LSTM (hidden=256) trained for 15 epochs with early stopping. Training used class-balanced upsampling across 10 price bins.

### Backtesting Results (3% deviation threshold, $1,000 starting capital)

| Run | Avg Final Value | Win Rate |
|---|---|---|
| LSTM — val markets | $1,092.83 | 3/12 profitable |
| LSTM — test markets | $1,016.15 | 4/27 profitable |
| Naive baseline — test | $930.11 | 8/27 profitable |

While we did see a profit in the test markets of ~$400, after 35 markets, the % profitablility is clearly a red flag with my current LSTM model that I will continue to look into. In addition, the error visualizations in the LSTModel, shows high error near the edges (0, 100) showing possible lack of data/other possible issues near the fringes that make its predictions worse. This could mean I should use different models near the edges, or avoid them all together to improve predictions/profit.

---

## Project Relevance
I looked into research paper's that address this very issue. Can/which ML models can actually be used to profit off statistical arbitrage?
https://link.springer.com/article/10.1007/s10614-021-10169-8

---

## Individual Contributions

This project was completed individually.

For a detailed breakdown of which functions were written, edited, or debugged with AI assistance, see ATTRIBUTION.md.
