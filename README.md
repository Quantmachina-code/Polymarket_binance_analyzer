# Step-by-step Binance + Polymarket Dataset Pipeline

This version uses **hardcoded slug-iteration logic** for Polymarket markets:

- `btc-updown-5m-<epoch>`
- `eth-updown-5m-<epoch>`
- `sol-updown-5m-<epoch>`

(and `bitcoin/ethereum/solana` variants).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Small setup (one crypto, quickest validation)

```bash
# 1) Binance BTC only
python pipeline.py step1-binance --asset btc --months 1 --output-dir data_small

# 2) Polymarket via hardcoded slug iteration (NOT discovery)
python pipeline.py step2-polymarket --asset btc --max-markets 10 --lookback-days 7 --output-dir data_small

# 3) Join + checks
python pipeline.py step3-join --asset btc --output-dir data_small
```

## Full setup

```bash
python pipeline.py run-steps --asset all --months 3 --max-markets 25 --lookback-days 14 --output-dir data
```

## Notes

- Step2 no longer discovers from generic market scans; it iterates expected slug names across 5-minute epoch buckets and fetches exact matches.
- Parquet dependencies are included in `requirements.txt` (`pyarrow`, `fastparquet`).
