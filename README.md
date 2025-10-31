# 🤖 Professional Forex Trading Bot with Machine Learning

A production-ready, fully automated forex trading system that combines institutional smart money concepts (Break of Structure, Fair Value Gaps), technical indicators, and machine learning predictions for MetaTrader 5.

## 🌟 Features

### Core Capabilities
- **Hybrid Trading Logic**: Rule-based confluence filters + XGBoost ML predictions
- **Institutional Concepts**: Fair Value Gaps (FVG) and Break of Structure (BOS) detection
- **Technical Analysis**: EMA(50/200), RSI(14), Stochastic(14,3,3), OBV, ATR(14)
- **Advanced Risk Management**: Daily/weekly limits, drawdown caps, position sizing, cooldown periods
- **Live Execution**: Automated trading on MetaTrader 5
- **Backtesting**: Walk-forward simulation with realistic costs
- **Real-time Monitoring**: Performance dashboards, drift detection, alerts

### Machine Learning
- **Model**: XGBoost classifier with hyperparameter optimization (Optuna)
- **Features**: 35+ engineered features including price action, momentum, volume, and time-based
- **Training**: Time-series cross-validation with proper chronological splits
- **Evaluation**: AUC, accuracy, precision, recall, Brier score, profit expectancy

### Risk Management
- **Position Sizing**: ATR-based with configurable risk per trade
- **Stop Loss & Take Profit**: Dynamic 1:2 R:R ratio using ATR
- **Safety Limits**: 
  - Daily loss limit: 3%
  - Weekly loss limit: 10%
  - Max drawdown: 20%
  - Loss streak cooldown: 2 hours after 3 consecutive losses
- **Volatility Filter**: Skip trades during low ATR periods

## 📁 Project Structure

```
forex_bot/
├── data_handler.py          # MT5 data collection & management
├── feature_engineering.py   # Technical indicators & features
├── label_generator.py       # Supervised learning labels
├── model_trainer.py         # XGBoost training & optimization
├── backtester.py           # Walk-forward backtesting
├── executor.py             # Live trade execution
├── risk_manager.py         # Risk controls & limits
├── monitor.py              # Performance monitoring & alerts
├── utils.py                # Helper functions & config
├── main.py                 # Main entry point
├── config.yaml             # Configuration file
└── README.md               # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.10+
- MetaTrader 5 terminal installed
- MT5 account (demo or live)

### Install Dependencies

```bash
pip install MetaTrader5 pandas numpy scikit-learn xgboost joblib ta matplotlib optuna scipy pyyaml
```

### Optional (for alerts)
```bash
pip install requests  # For Telegram alerts
```

## 🚀 Quick Start

### 1. Configure the Bot

Create or edit `config.yaml`:

```yaml
mt5:
  path: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
  symbols: ['EURUSD', 'GBPUSD', 'USDJPY']
  timeframe: '15m'

risk:
  initial_balance: 10000
  risk_per_trade: 0.01  # 1%
  daily_loss_limit: 0.03  # 3%
  max_drawdown_limit: 0.20  # 20%

trading:
  sl_atr_mult: 1.0
  tp_atr_mult: 2.0
  prob_threshold: 0.6
  check_interval: 900  # 15 minutes
```

### 2. Download Historical Data

```bash
python main.py download --symbol EURUSD --days 730
```

### 3. Train ML Model

```bash
# Quick training with default parameters
python main.py train --symbol EURUSD

# With hyperparameter optimization (recommended)
python main.py train --symbol EURUSD --optimize --trials 100
```

### 4. Backtest Strategy

```bash
# Backtest with ML model
python main.py backtest --symbol EURUSD

# Backtest rule-based only
python main.py backtest --symbol EURUSD --no-model
```

### 5. Run Live Trading

```bash
python main.py live --symbol EURUSD
```

## 📊 Trading Logic

### Entry Conditions

A trade is executed when **at least 3 out of 5** conditions are met:

**Long Trade:**
1. ✅ EMA50 > EMA200 (uptrend)
2. ✅ Price pullback or RSI/Stoch oversold
3. ✅ Stochastic crossover up
4. ✅ Bullish FVG or BOS detected
5. ✅ RSI not overbought (< 70)

**Short Trade:**
1. ✅ EMA50 < EMA200 (downtrend)
2. ✅ Price pullback or RSI/Stoch overbought
3. ✅ Stochastic crossover down
4. ✅ Bearish FVG or BOS detected
5. ✅ RSI not oversold (> 30)

**Plus:** ML model probability must be > 0.6 (configurable)

### Exit Strategy
- **Take Profit**: 2× ATR from entry
- **Stop Loss**: 1× ATR from entry
- **Time Exit**: Maximum 48 bars (12 hours on 15m chart)

## 🎯 Feature Engineering

### Technical Indicators
- **Trend**: EMA(50), EMA(200), slopes, spreads
- **Momentum**: RSI(14), Stochastic(14,3,3)
- **Volume**: OBV, OBV delta (24h)
- **Volatility**: ATR(14)

### Institutional Concepts
- **Fair Value Gaps (FVG)**: Price imbalances > 0.5× ATR
- **Break of Structure (BOS)**: New swing highs/lows breaking previous structure

### Derived Features
- Price-to-EMA distance (ATR-normalized)
- EMA slopes and crossovers
- RSI divergence flags
- Time-based features (hour, day of week - cyclically encoded)
- Trading session indicators (Asian, London, NY)

## 📈 Performance Metrics

The bot tracks and reports:
- **Profit Metrics**: Total return, daily/weekly P&L, profit factor
- **Risk Metrics**: Max drawdown, Sharpe ratio, Sortino ratio
- **Trade Metrics**: Win rate, expectancy, average win/loss
- **ML Metrics**: AUC-ROC, accuracy, precision, recall, Brier score

## 🔒 Safety Features

### Automatic Shutdown
The bot stops trading if:
- Daily loss exceeds 3%
- Weekly loss exceeds 10%
- Drawdown exceeds 20%
- 3 consecutive losses (2-hour cooldown)

### Position Limits
- Max 2 positions per symbol
- Max 5 total open positions
- Volatility filter (minimum ATR requirement)

### Error Handling
- Graceful MT5 connection failures
- Data feed validation
- Trade execution retries
- Comprehensive logging

## 📧 Monitoring & Alerts

### Dashboard
Real-time plots showing:
- Equity curve
- Drawdown
- Win rate
- Daily P&L
- Trade count
- Profit factor

### Alerts
Configure email or Telegram alerts for:
- Daily loss approaching limit
- High drawdown warnings
- Trading halted events
- Feature drift detection
- Execution failures

## 🧪 Backtesting

The backtester includes:
- Realistic transaction costs (spread, commission, slippage)
- Time-series walk-forward validation
- Detailed trade log with entry/exit reasons
- Performance visualization
- Monte Carlo robustness testing

## 📝 Command Reference

### Download Data
```bash
python main.py download --symbol EURUSD --days 365
```

### Train Model
```bash
# Default parameters
python main.py train --symbol EURUSD

# With optimization
python main.py train --symbol EURUSD --optimize --trials 50
```

### Backtest
```bash
# With ML model
python main.py backtest --symbol EURUSD

# Rules only
python main.py backtest --symbol EURUSD --no-model
```

### Live Trading
```bash
python main.py live --symbol EURUSD
```

### Custom Config
```bash
python main.py train --symbol GBPUSD --config custom_config.yaml
```

## ⚠️ Risk Disclaimer

**THIS SOFTWARE IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

- Trading forex carries substantial risk of loss
- Past performance does not guarantee future results
- Never trade with money you cannot afford to lose
- Always test thoroughly on a demo account first
- The authors assume no liability for financial losses

## 🛠️ Customization

### Adjust Risk Parameters
Edit `config.yaml` or modify `risk_manager.py`:
- Risk per trade percentage
- Daily/weekly loss limits
- Max drawdown threshold
- Cooldown periods

### Modify Trading Logic
Edit `executor.py` method `check_rule_based_conditions()`:
- Add/remove technical indicators
- Adjust signal thresholds
- Change confluence requirements

### Enhance ML Model
Edit `model_trainer.py`:
- Try different algorithms (LightGBM, Random Forest)
- Add more features in `feature_engineering.py`
- Adjust hyperparameter search space

## 📚 Advanced Usage

### Walk-Forward Optimization
```python
from backtester import Backtester

# Implement rolling window backtest
# Train on 1 year, test on 3 months, roll forward
```

### Multi-Symbol Trading
```python
from executor import TradeExecutor

# Run multiple executor instances for different pairs
# Manage cross-symbol risk limits
```

### Custom Indicators
```python
from feature_engineering import FeatureEngineer

# Add your own indicators in FeatureEngineer class
# Update feature list for ML training
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional technical indicators
- More sophisticated ML models (LSTM, Transformer)
- Multi-timeframe analysis
- Correlation-based pair trading
- Advanced order types (trailing stops, scaled entries)

## 📖 References

- **Smart Money Concepts**: ICT (Inner Circle Trader)
- **Machine Learning**: XGBoost Documentation
- **Risk Management**: "Trade Your Way to Financial Freedom" by Van K. Tharp
- **MetaTrader 5**: Official MT5 Python Documentation

## 📞 Support

For issues, questions, or suggestions:
- Check the code comments (extensively documented)
- Review the configuration file
- Test on demo account first
- Verify MT5 connection settings

## 📜 License

MIT License - See LICENSE file for details

---

**Built with ❤️ for quantitative traders**

*Remember: Discipline, risk management, and continuous learning are the keys to success in algorithmic trading.*