# Step-by-step Binance + Dynamic Polymarket Dataset Pipeline

This pipeline is designed to be testable in small runs first:

1. **Step 1**: Download Binance 1-minute data + quality report.
2. **Step 2**: Discover dynamic Polymarket 5m up/down markets + fetch history + quality report.
3. **Step 3**: Join both datasets + quality report.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Small setup (one crypto, quickest validation)

```bash
# BTC only on Binance
python pipeline.py step1-binance --asset btc --months 1 --output-dir data_small

# BTC dynamic Polymarket discovery, max 10 markets, last 7 days
python pipeline.py step2-polymarket --asset btc --max-markets 10 --lookback-days 7 --output-dir data_small

# Join + checks
python pipeline.py step3-join --asset btc --output-dir data_small
```

## Full setup (all cryptos)

```bash
python pipeline.py run-steps --asset all --months 3 --max-markets 25 --lookback-days 14 --output-dir data
```

## Outputs

- `binance_1m.parquet`
- `binance_quality_report.csv`
- `polymarket_dynamic_markets.parquet`
- `polymarket_history.parquet`
- `polymarket_orderbook_snapshots.parquet`
- `polymarket_quality_report.csv`
- `joined_modeling_dataset.parquet`
- `joined_quality_report.csv`

## Notes

- Dynamic Polymarket market discovery is done from Gamma `/markets` data and filtered by rolling 5m up/down patterns (e.g. `btc-updown-5m-...`, `sol-updown-5m-...`).
- `pyarrow` and `fastparquet` are included in requirements for parquet support.
