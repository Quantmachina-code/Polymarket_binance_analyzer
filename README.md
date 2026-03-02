# Polymarket + Binance Crypto Prediction Pipeline

This repository provides a step-by-step, testable pipeline to:

1. Download 1-minute Binance OHLCV data (BTCUSDC, ETHUSDC, SOLUSDC) for the last 3 months.
2. Dynamically discover Polymarket historical data endpoints and download 1-minute + 5-minute market data for crypto markets (and optional 15-minute where available).
3. Build a comprehensive modeling dataset with technical indicators, volume features, lags, and Polymarket microstructure-style features.
4. Train/evaluate a probabilistic model for the next 5-minute market move.
5. Simulate a simple arbitrage-style trading strategy using model probabilities vs Polymarket implied probabilities.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py run-all --months 3 --output-dir data
```

## Stage-by-stage commands

```bash
# 1) Binance data only
python pipeline.py fetch-binance --months 3 --output-dir data

# 2) Discover Polymarket hosts + fetch history/orderbook snapshots
python pipeline.py fetch-polymarket --output-dir data

# 3) Build features and targets
python pipeline.py build-features --output-dir data

# 4) Train + evaluate
python pipeline.py train-model --output-dir data

# 5) Backtest trading/arbitrage logic
python pipeline.py backtest --output-dir data
```

## Notes

- Polymarket APIs can evolve. Host and route discovery is implemented dynamically by probing known hosts/routes.
- For historical orderbook depth, public endpoints may be limited. The pipeline stores available orderbook snapshots and trade/price history to approximate buy/sell pressure features.
- The model is probability-oriented and calibrated with isotonic regression.
