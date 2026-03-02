import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


BINANCE_BASE = "https://api.binance.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
SYMBOLS = ["BTCUSDC", "ETHUSDC", "SOLUSDC"]
EVENT_PATTERNS = [
    re.compile(r"btc-updown-5m", re.IGNORECASE),
    re.compile(r"eth-updown-5m", re.IGNORECASE),
    re.compile(r"sol-updown-5m", re.IGNORECASE),
    re.compile(r"bitcoin-updown-5m", re.IGNORECASE),
    re.compile(r"ethereum-updown-5m", re.IGNORECASE),
    re.compile(r"solana-updown-5m", re.IGNORECASE),
]


@dataclass
class EndpointCandidate:
    base_url: str
    path: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _request_json(url: str, params: Optional[dict] = None, timeout: int = 30):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


# ===========================
# STEP 1: BINANCE + QA SWEEP
# ===========================
def fetch_binance_1m(symbol: str, months: int) -> pd.DataFrame:
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=30 * months)

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    rows: List[list] = []
    while start_ms < end_ms:
        payload = _request_json(
            f"{BINANCE_BASE}/api/v3/klines",
            {
                "symbol": symbol,
                "interval": "1m",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not payload:
            break

        rows.extend(payload)
        start_ms = int(payload[-1][0]) + 60_000
        time.sleep(0.05)

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce").astype("Int64")

    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["symbol"] = symbol

    return df[["ts", "symbol", "open", "high", "low", "close", "volume", "quote_asset_volume", "num_trades", "taker_buy_base", "taker_buy_quote"]].sort_values("ts")


def binance_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    reports = []
    for symbol, g in df.groupby("symbol"):
        g = g.sort_values("ts").reset_index(drop=True)
        diffs = g["ts"].diff().dt.total_seconds().dropna()
        missing_minutes = int((diffs > 60).sum())

        report = {
            "symbol": symbol,
            "rows": int(len(g)),
            "start_ts": g["ts"].min(),
            "end_ts": g["ts"].max(),
            "duplicate_timestamps": int(g.duplicated(subset=["ts"]).sum()),
            "missing_minutes_gaps": missing_minutes,
            "null_close": int(g["close"].isna().sum()),
            "non_positive_price_rows": int((g["close"] <= 0).sum()),
            "non_positive_volume_rows": int((g["volume"] < 0).sum()),
        }
        reports.append(report)

    return pd.DataFrame(reports)


# ========================================
# STEP 2: DYNAMIC POLYMARKET + QA SWEEP
# ========================================
def discover_polymarket_history_endpoints() -> List[EndpointCandidate]:
    candidates = [
        EndpointCandidate(CLOB_BASE, "/prices-history"),
        EndpointCandidate(CLOB_BASE, "/price-history"),
        EndpointCandidate("https://data-api.polymarket.com", "/prices-history"),
        EndpointCandidate("https://data-api.polymarket.com", "/price-history"),
    ]

    working: List[EndpointCandidate] = []
    probe_params = {
        "market": "0",
        "startTs": int(time.time()) - 3600,
        "endTs": int(time.time()),
        "fidelity": 300,
    }

    for c in candidates:
        try:
            r = requests.get(f"{c.base_url}{c.path}", params=probe_params, timeout=7)
            if r.status_code != 404:
                working.append(c)
        except requests.RequestException:
            pass

    return working


def _is_target_event_slug(slug: str) -> bool:
    if not slug:
        return False
    return any(pattern.search(slug) for pattern in EVENT_PATTERNS)


def fetch_dynamic_polymarket_events(limit: int = 1000) -> pd.DataFrame:
    events = _request_json(f"{GAMMA_BASE}/events", {"limit": limit, "active": True, "closed": False})
    df = pd.DataFrame(events)
    if df.empty:
        return df

    slug_col = "slug" if "slug" in df.columns else None
    if slug_col is None:
        return pd.DataFrame()

    target = df[df[slug_col].fillna("").apply(_is_target_event_slug)].copy()
    return target


def _extract_token_ids(market_row: pd.Series) -> List[str]:
    for key in ["clobTokenIds", "clob_token_ids"]:
        if key in market_row and pd.notna(market_row[key]):
            raw = market_row[key]
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    return [str(x) for x in parsed]
                except Exception:
                    return []
            if isinstance(raw, list):
                return [str(x) for x in raw]
    return []


def fetch_event_markets(event_ids: List[str], market_limit: int = 2000) -> pd.DataFrame:
    markets = _request_json(f"{GAMMA_BASE}/markets", {"limit": market_limit, "active": True, "closed": False})
    mdf = pd.DataFrame(markets)
    if mdf.empty:
        return mdf

    event_key = "eventId" if "eventId" in mdf.columns else "event_id"
    if event_key not in mdf.columns:
        return pd.DataFrame()

    wanted = mdf[mdf[event_key].astype(str).isin(set(event_ids))].copy()
    return wanted


def fetch_polymarket_history(markets_df: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    endpoints = discover_polymarket_history_endpoints()

    now = int(time.time())
    start = now - 90 * 24 * 3600

    hist_rows = []
    orderbook_rows = []

    for _, row in markets_df.iterrows():
        market_id = str(row.get("id", row.get("conditionId", "")))
        token_ids = _extract_token_ids(row)
        if not market_id:
            continue

        for fidelity in [60, 300]:
            for ep in endpoints:
                try:
                    payload = _request_json(
                        f"{ep.base_url}{ep.path}",
                        {"market": market_id, "startTs": start, "endTs": now, "fidelity": fidelity},
                        timeout=10,
                    )
                    history = payload.get("history", payload if isinstance(payload, list) else [])
                    for point in history:
                        ts = point.get("t") or point.get("timestamp")
                        price = point.get("p") or point.get("price")
                        if ts is None or price is None:
                            continue
                        hist_rows.append(
                            {
                                "market_id": market_id,
                                "fidelity": fidelity,
                                "ts": pd.to_datetime(int(ts), unit="s", utc=True),
                                "price": float(price),
                            }
                        )
                    break
                except Exception:
                    continue

        for token_id in token_ids:
            try:
                ob = _request_json(f"{CLOB_BASE}/book", {"token_id": token_id}, timeout=8)
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                orderbook_rows.append(
                    {
                        "market_id": market_id,
                        "token_id": token_id,
                        "snapshot_ts": pd.Timestamp.utcnow(),
                        "best_bid": float(bids[0]["price"]) if bids else None,
                        "best_ask": float(asks[0]["price"]) if asks else None,
                        "bid_size": float(bids[0]["size"]) if bids else 0.0,
                        "ask_size": float(asks[0]["size"]) if asks else 0.0,
                    }
                )
            except Exception:
                continue

    hist_df = pd.DataFrame(hist_rows)
    ob_df = pd.DataFrame(orderbook_rows)
    hist_df.to_parquet(out_dir / "polymarket_history.parquet", index=False)
    ob_df.to_parquet(out_dir / "polymarket_orderbook_snapshots.parquet", index=False)

    return hist_df, ob_df


def polymarket_quality_report(hist_df: pd.DataFrame, ob_df: pd.DataFrame) -> pd.DataFrame:
    if hist_df.empty:
        return pd.DataFrame()

    rows = []
    for fidelity, g in hist_df.groupby("fidelity"):
        g = g.sort_values(["market_id", "ts"])
        dupes = int(g.duplicated(subset=["market_id", "ts"]).sum())
        null_price = int(g["price"].isna().sum())
        invalid_price = int(((g["price"] < 0) | (g["price"] > 1)).sum())

        rows.append(
            {
                "fidelity": int(fidelity),
                "rows": int(len(g)),
                "unique_markets": int(g["market_id"].nunique()),
                "duplicates_market_ts": dupes,
                "null_price": null_price,
                "price_out_of_0_1": invalid_price,
                "orderbook_snapshots": int(len(ob_df)),
            }
        )

    return pd.DataFrame(rows)


# ====================================
# STEP 3: JOIN + QA SWEEP
# ====================================
def build_joined_dataset(binance_df: pd.DataFrame, polymarket_hist_df: pd.DataFrame) -> pd.DataFrame:
    b = binance_df.copy().sort_values("ts")

    # resample Binance to 5m for easier alignment with up/down 5m events
    b5 = (
        b.set_index("ts")
        .groupby("symbol")
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_asset_volume": "sum", "num_trades": "sum"})
        .reset_index()
    )

    # pivot symbols to one row per timestamp
    close_wide = b5.pivot(index="ts", columns="symbol", values="close").rename(columns=lambda c: f"{c}_close")
    volume_wide = b5.pivot(index="ts", columns="symbol", values="volume").rename(columns=lambda c: f"{c}_volume")
    features = close_wide.join(volume_wide).reset_index()

    # Use 5m polymarket history for target horizon
    pm5 = polymarket_hist_df[polymarket_hist_df["fidelity"] == 300].copy()
    pm5 = pm5.sort_values(["market_id", "ts"])
    pm5["target_up_5m"] = (pm5.groupby("market_id")["price"].shift(-1) > pm5["price"]).astype(int)

    target = pm5.groupby("ts", as_index=False).agg(pm_yes_price_mean=("price", "mean"), target_up_5m=("target_up_5m", "mean"))
    target["target_up_5m"] = (target["target_up_5m"] > 0.5).astype(int)

    joined = target.merge(features, on="ts", how="left").sort_values("ts")

    for c in [col for col in joined.columns if col not in ["ts", "target_up_5m"]]:
        joined[f"lag1_{c}"] = joined[c].shift(1)

    joined = joined.dropna().reset_index(drop=True)
    return joined


def joined_quality_report(joined_df: pd.DataFrame) -> pd.DataFrame:
    if joined_df.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "rows": int(len(joined_df)),
                "start_ts": joined_df["ts"].min(),
                "end_ts": joined_df["ts"].max(),
                "null_cells": int(joined_df.isna().sum().sum()),
                "duplicate_ts": int(joined_df.duplicated(subset=["ts"]).sum()),
                "target_up_rate": float(joined_df["target_up_5m"].mean()),
            }
        ]
    )


def cmd_step1_binance(args):
    out = Path(args.output_dir)
    ensure_dir(out)

    chunks = []
    for symbol in SYMBOLS:
        print(f"Downloading Binance 1m: {symbol}")
        chunks.append(fetch_binance_1m(symbol=symbol, months=args.months))

    all_binance = pd.concat(chunks, ignore_index=True)
    all_binance.to_parquet(out / "binance_1m.parquet", index=False)

    report = binance_quality_report(all_binance)
    report.to_csv(out / "binance_quality_report.csv", index=False)
    print(report)


def cmd_step2_polymarket(args):
    out = Path(args.output_dir)
    ensure_dir(out)

    events_df = fetch_dynamic_polymarket_events(limit=args.event_limit)
    events_df.to_parquet(out / "polymarket_dynamic_events.parquet", index=False)

    event_ids = [str(x) for x in events_df.get("id", pd.Series([], dtype=str)).tolist()]
    markets_df = fetch_event_markets(event_ids=event_ids, market_limit=args.market_limit)
    markets_df.to_parquet(out / "polymarket_dynamic_markets.parquet", index=False)

    hist_df, ob_df = fetch_polymarket_history(markets_df, out)
    report = polymarket_quality_report(hist_df, ob_df)
    report.to_csv(out / "polymarket_quality_report.csv", index=False)
    print(report)


def cmd_step3_join(args):
    out = Path(args.output_dir)
    ensure_dir(out)

    binance_df = pd.read_parquet(out / "binance_1m.parquet")
    poly_hist_df = pd.read_parquet(out / "polymarket_history.parquet")

    joined = build_joined_dataset(binance_df, poly_hist_df)
    joined.to_parquet(out / "joined_modeling_dataset.parquet", index=False)

    report = joined_quality_report(joined)
    report.to_csv(out / "joined_quality_report.csv", index=False)
    print(report)


def cmd_run_steps(args):
    cmd_step1_binance(args)
    cmd_step2_polymarket(args)
    cmd_step3_join(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step-by-step Binance + dynamic Polymarket data pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=str, default="data")

    s1 = sub.add_parser("step1-binance", parents=[common], help="Download Binance 1m + quality checks")
    s1.add_argument("--months", type=int, default=3)
    s1.set_defaults(func=cmd_step1_binance)

    s2 = sub.add_parser("step2-polymarket", parents=[common], help="Dynamic Polymarket markets + historical fetch + quality checks")
    s2.add_argument("--event-limit", type=int, default=1000)
    s2.add_argument("--market-limit", type=int, default=2000)
    s2.set_defaults(func=cmd_step2_polymarket)

    s3 = sub.add_parser("step3-join", parents=[common], help="Join Binance and Polymarket + quality checks")
    s3.set_defaults(func=cmd_step3_join)

    s4 = sub.add_parser("run-steps", parents=[common], help="Run steps 1 -> 3")
    s4.add_argument("--months", type=int, default=3)
    s4.add_argument("--event-limit", type=int, default=1000)
    s4.add_argument("--market-limit", type=int, default=2000)
    s4.set_defaults(func=cmd_run_steps)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
