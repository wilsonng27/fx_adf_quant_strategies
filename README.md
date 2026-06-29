# ADF Quantitative FX Trading Strategies 📈

This repository contains a suite of algorithmic trading backtest scripts focusing on mean-reversion strategies applied to major FX pairs (EURUSD, GBPUSD, USDCAD, USDJPY). 

A core component of all three strategies is the use of the **Augmented Dickey-Fuller (ADF) test**. Trades are only executed when the statistical test confirms the market is currently in a mean-reverting (stationary) regime, filtering out strong trending periods where mean-reversion tends to fail.

## Strategies Included

1. **`01_bollinger_reversion.py`**
   Uses Bollinger Bands to define overbought/oversold levels. 

2. **`02_price_action.py`**
   Focuses on consecutive down/up days (streaks) combined with percentile ranking within a recent rolling price range. 

3. **`03_rsi_reversion.py`**
   Utilizes the highly reactive Connors RSI-2 indicator. It enters long on extreme RSI lows (<10) and short on extreme RSI highs (>90), provided the ADF test confirms stationarity.

## Installation & Setup

To run these backtests locally, clone the repository and install the dependencies:

```bash
git clone [https://github.com/YOUR-USERNAME-HERE/fx-adf-quant-strategies.git](https://github.com/YOUR-USERNAME-HERE/fx-adf-quant-strategies.git)
cd fx-adf-quant-strategies
pip install -r requirements.txt
