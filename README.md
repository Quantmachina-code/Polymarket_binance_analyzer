# Step-by-step Binance + Dynamic Polymarket Dataset Pipeline

This project now follows the exact staged workflow:

1. **Step 1**: Download Binance 1-minute data for `BTCUSDC`, `ETHUSDC`, `SOLUSDC` (default past 3 months), then run a data-quality sweep.
2. **Step 2**: Dynamically discover active Polymarket up/down 5-minute crypto events (e.g. rolling URLs like `sol-updown-5m-...`, `btc-updown-5m-...`), fetch history/orderbook snapshots, then run quality checks.
3. **Step 3**: Join Binance + Polymarket datasets into a modeling-ready table and run join quality checks.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run stage by stage

```bash
# Step 1: Binance 1m + quality
python pipeline.py step1-binance --months 3 --output-dir data

# Step 2: Dynamic Polymarket discovery + history/orderbook + quality
python pipeline.py step2-polymarket --output-dir data

# Step 3: Join and quality sweep
python pipeline.py step3-join --output-dir data
```

Or all three:

```bash
python pipeline.py run-steps --months 3 --output-dir data
```

## Output files

- `data/binance_1m.parquet`
- `data/binance_quality_report.csv`
- `data/polymarket_dynamic_events.parquet`
- `data/polymarket_dynamic_markets.parquet`
- `data/polymarket_history.parquet`
- `data/polymarket_orderbook_snapshots.parquet`
- `data/polymarket_quality_report.csv`
- `data/joined_modeling_dataset.parquet`
- `data/joined_quality_report.csv`

## Notes

- Polymarket event URLs are dynamic over time; this pipeline detects matching active events by slug pattern instead of hardcoding a fixed URL.
- If an endpoint changes behavior, endpoint probing is used to find a working historical route.
