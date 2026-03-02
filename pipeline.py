import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score


BINANCE_BASE = "https://api.binance.com"
SYMBOLS = ["BTCUSDC", "ETHUSDC", "SOLUSDC"]


@dataclass
class EndpointCandidate:
    base_url: str
    path: str


def _request_json(url: str, params: Optional[dict] = None, timeout: int = 20):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fetch_binance_1m(symbol: str, months: int = 3) -> pd.DataFrame:
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=30 * months)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_rows = []
    while start_ms < end_ms:
        data = _request_json(
            f"{BINANCE_BASE}/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": "1m",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not data:
            break
        all_rows.extend(data)
        start_ms = int(data[-1][0]) + 60_000
        time.sleep(0.06)

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
    df = pd.DataFrame(all_rows, columns=cols)
    if df.empty:
        return df

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["symbol"] = symbol
    return df[["ts", "symbol", "open", "high", "low", "close", "volume", "num_trades", "taker_buy_base"]]


def discover_polymarket_history_endpoints() -> List[EndpointCandidate]:
    candidates = [
        EndpointCandidate("https://clob.polymarket.com", "/prices-history"),
        EndpointCandidate("https://clob.polymarket.com", "/price-history"),
        EndpointCandidate("https://data-api.polymarket.com", "/prices-history"),
        EndpointCandidate("https://data-api.polymarket.com", "/price-history"),
    ]

    working = []
    probe_market = "0"
    now = int(time.time())
    params = {"market": probe_market, "startTs": now - 3600, "endTs": now, "fidelity": 60}

    for c in candidates:
        try:
            requests.get(f"{c.base_url}{c.path}", params=params, timeout=6)
            # if endpoint exists, we'll accept any non-404 response (schema may vary by market id)
            resp = requests.get(f"{c.base_url}{c.path}", params=params, timeout=6)
            if resp.status_code != 404:
                working.append(c)
        except requests.RequestException:
            continue
    return working


def fetch_crypto_markets(limit: int = 500) -> pd.DataFrame:
    # Gamma API catalog is usually the best market discovery source.
    rows = _request_json(
        "https://gamma-api.polymarket.com/markets",
        params={"limit": limit, "active": True, "closed": False},
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    search_cols = [c for c in ["question", "description", "slug"] if c in df.columns]
    text = df[search_cols].fillna("").agg(" ".join, axis=1).str.lower() if search_cols else pd.Series("", index=df.index)
    mask = text.str.contains("bitcoin|btc|ethereum|eth|solana|sol|crypto", regex=True)
    return df.loc[mask].copy()


def _extract_token_ids(market_row: pd.Series) -> List[str]:
    token_ids = []
    for key in ["clobTokenIds", "clob_token_ids"]:
        if key in market_row and pd.notna(market_row[key]):
            try:
                parsed = json.loads(market_row[key]) if isinstance(market_row[key], str) else market_row[key]
                token_ids.extend([str(x) for x in parsed])
            except Exception:
                pass
    return list(dict.fromkeys(token_ids))


def fetch_polymarket_market_data(markets: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    endpoints = discover_polymarket_history_endpoints()
    now = int(time.time())
    start = now - 90 * 24 * 3600

    hist_rows = []
    orderbook_rows = []

    for _, row in markets.iterrows():
        market_id = str(row.get("id", row.get("conditionId", "")))
        token_ids = _extract_token_ids(row)
        if not market_id or not token_ids:
            continue

        for fidelity in [60, 300, 900]:
            for ep in endpoints:
                try:
                    payload = _request_json(
                        f"{ep.base_url}{ep.path}",
                        params={"market": market_id, "startTs": start, "endTs": now, "fidelity": fidelity},
                        timeout=10,
                    )
                    series = payload.get("history", payload if isinstance(payload, list) else [])
                    for p in series:
                        ts = p.get("t") or p.get("timestamp")
                        price = p.get("p") or p.get("price")
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

        # best-effort current orderbook snapshots for YES/NO tokens
        for t in token_ids:
            try:
                b = _request_json("https://clob.polymarket.com/book", params={"token_id": t}, timeout=8)
                bids = b.get("bids", [])
                asks = b.get("asks", [])
                best_bid = float(bids[0]["price"]) if bids else np.nan
                best_ask = float(asks[0]["price"]) if asks else np.nan
                bid_size = float(bids[0]["size"]) if bids else 0.0
                ask_size = float(asks[0]["size"]) if asks else 0.0
                orderbook_rows.append(
                    {
                        "market_id": market_id,
                        "token_id": t,
                        "snapshot_ts": pd.Timestamp.utcnow(),
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "bid_size": bid_size,
                        "ask_size": ask_size,
                    }
                )
            except Exception:
                continue

    hist_df = pd.DataFrame(hist_rows)
    ob_df = pd.DataFrame(orderbook_rows)
    hist_df.to_parquet(out_dir / "polymarket_history.parquet", index=False)
    ob_df.to_parquet(out_dir / "polymarket_orderbook_snapshots.parquet", index=False)
    return hist_df, ob_df


def add_technical_features(df: pd.DataFrame, price_col: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_ret_1"] = out[price_col].pct_change(1)
    out[f"{prefix}_ret_5"] = out[price_col].pct_change(5)
    out[f"{prefix}_ret_15"] = out[price_col].pct_change(15)
    out[f"{prefix}_vol_15"] = out[f"{prefix}_ret_1"].rolling(15).std()
    out[f"{prefix}_ma_5"] = out[price_col].rolling(5).mean()
    out[f"{prefix}_ma_20"] = out[price_col].rolling(20).mean()
    out[f"{prefix}_ma_cross"] = out[f"{prefix}_ma_5"] / out[f"{prefix}_ma_20"] - 1
    return out


def build_modeling_dataset(binance_df: pd.DataFrame, poly_hist_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    ensure_dir(out_dir)

    b = binance_df.copy().sort_values("ts")
    frames = []
    for sym, g in b.groupby("symbol"):
        gi = add_technical_features(g, "close", sym.lower())
        gi = gi[[c for c in gi.columns if c.startswith(sym.lower()) or c in ["ts"]]]
        frames.append(gi)

    merged = frames[0]
    for fr in frames[1:]:
        merged = merged.merge(fr, on="ts", how="outer")

    # Prefer 5-minute fidelity for target market probability dynamics.
    ph = poly_hist_df[poly_hist_df["fidelity"] == 300].copy()
    ph = ph.sort_values(["market_id", "ts"])
    ph["pm_ret_1"] = ph.groupby("market_id")["price"].pct_change(1)
    ph["pm_ret_3"] = ph.groupby("market_id")["price"].pct_change(3)
    ph["target_up_5m"] = (ph.groupby("market_id")["price"].shift(-1) > ph["price"]).astype(int)

    # aggregate across active markets at each timestamp
    agg = (
        ph.groupby("ts")
        .agg(pm_price_mean=("price", "mean"), pm_price_std=("price", "std"), pm_ret_1_mean=("pm_ret_1", "mean"), target_up_5m=("target_up_5m", "mean"))
        .reset_index()
    )
    agg["target_up_5m"] = (agg["target_up_5m"] > 0.5).astype(int)

    df = agg.merge(merged, on="ts", how="left").sort_values("ts")

    for col in [c for c in df.columns if c not in ["ts", "target_up_5m"]]:
        df[f"lag1_{col}"] = df[col].shift(1)
        df[f"lag3_{col}"] = df[col].shift(3)

    df = df.dropna().reset_index(drop=True)
    df.to_parquet(out_dir / "modeling_dataset.parquet", index=False)
    return df


def train_and_evaluate(df: pd.DataFrame, out_dir: Path) -> Dict[str, float]:
    ensure_dir(out_dir)
    df = df.sort_values("ts").reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in ["ts", "target_up_5m"]]
    X = df[feat_cols]
    y = df["target_up_5m"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    base = RandomForestClassifier(n_estimators=300, random_state=42, min_samples_leaf=8, n_jobs=-1)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "test_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else np.nan,
        "test_brier": float(brier_score_loss(y_test, proba)),
        "test_accuracy": float((preds == y_test).mean()),
    }

    pred_df = df.iloc[split:][["ts"]].copy()
    pred_df["y_true"] = y_test.values
    pred_df["p_up"] = proba
    pred_df.to_parquet(out_dir / "test_predictions.parquet", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def backtest_arbitrage(pred_df: pd.DataFrame, market_df: pd.DataFrame, out_dir: Path, edge: float = 0.03) -> pd.DataFrame:
    ensure_dir(out_dir)
    m = market_df[market_df["fidelity"] == 300].groupby("ts", as_index=False)["price"].mean().rename(columns={"price": "pm_yes_price"})

    bt = pred_df.merge(m, on="ts", how="inner").sort_values("ts")
    bt["pm_no_price"] = 1 - bt["pm_yes_price"]

    # Simple edge logic: if model p_up > yes_price + edge, buy YES; if below yes_price - edge, buy NO.
    bt["position"] = 0
    bt.loc[bt["p_up"] > bt["pm_yes_price"] + edge, "position"] = 1
    bt.loc[bt["p_up"] < bt["pm_yes_price"] - edge, "position"] = -1

    bt["realized"] = np.where(bt["y_true"] == 1, 1.0, 0.0)
    bt["trade_pnl"] = 0.0
    yes_idx = bt["position"] == 1
    no_idx = bt["position"] == -1
    bt.loc[yes_idx, "trade_pnl"] = bt.loc[yes_idx, "realized"] - bt.loc[yes_idx, "pm_yes_price"]
    bt.loc[no_idx, "trade_pnl"] = (1 - bt.loc[no_idx, "realized"]) - bt.loc[no_idx, "pm_no_price"]

    bt["cum_pnl"] = bt["trade_pnl"].cumsum()
    bt.to_parquet(out_dir / "backtest_results.parquet", index=False)
    return bt


def cmd_fetch_binance(args):
    out = Path(args.output_dir)
    ensure_dir(out)
    dfs = []
    for sym in SYMBOLS:
        df = fetch_binance_1m(sym, months=args.months)
        dfs.append(df)
    result = pd.concat(dfs, ignore_index=True)
    result.to_parquet(out / "binance_1m.parquet", index=False)
    print(f"Saved {len(result):,} Binance rows")


def cmd_fetch_polymarket(args):
    out = Path(args.output_dir)
    ensure_dir(out)
    markets = fetch_crypto_markets(limit=args.market_limit)
    markets.to_parquet(out / "polymarket_crypto_markets.parquet", index=False)
    hist, ob = fetch_polymarket_market_data(markets, out)
    print(f"Saved polymarket history rows={len(hist):,}, orderbook snapshots={len(ob):,}")


def cmd_build_features(args):
    out = Path(args.output_dir)
    b = pd.read_parquet(out / "binance_1m.parquet")
    p = pd.read_parquet(out / "polymarket_history.parquet")
    ds = build_modeling_dataset(b, p, out)
    print(f"Saved modeling dataset rows={len(ds):,}")


def cmd_train_model(args):
    out = Path(args.output_dir)
    ds = pd.read_parquet(out / "modeling_dataset.parquet")
    metrics = train_and_evaluate(ds, out)
    print(json.dumps(metrics, indent=2))


def cmd_backtest(args):
    out = Path(args.output_dir)
    preds = pd.read_parquet(out / "test_predictions.parquet")
    market = pd.read_parquet(out / "polymarket_history.parquet")
    bt = backtest_arbitrage(preds, market, out, edge=args.edge)
    summary = {
        "trades": int((bt["position"] != 0).sum()),
        "avg_trade_pnl": float(bt.loc[bt["position"] != 0, "trade_pnl"].mean() if (bt["position"] != 0).any() else 0),
        "total_pnl": float(bt["trade_pnl"].sum()),
    }
    print(json.dumps(summary, indent=2))


def cmd_run_all(args):
    cmd_fetch_binance(args)
    cmd_fetch_polymarket(args)
    cmd_build_features(args)
    cmd_train_model(args)
    cmd_backtest(args)


def parser():
    p = argparse.ArgumentParser(description="Polymarket + Binance prediction/arbitrage pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=str, default="data")

    p1 = sub.add_parser("fetch-binance", parents=[common])
    p1.add_argument("--months", type=int, default=3)
    p1.set_defaults(func=cmd_fetch_binance)

    p2 = sub.add_parser("fetch-polymarket", parents=[common])
    p2.add_argument("--market-limit", type=int, default=500)
    p2.set_defaults(func=cmd_fetch_polymarket)

    p3 = sub.add_parser("build-features", parents=[common])
    p3.set_defaults(func=cmd_build_features)

    p4 = sub.add_parser("train-model", parents=[common])
    p4.set_defaults(func=cmd_train_model)

    p5 = sub.add_parser("backtest", parents=[common])
    p5.add_argument("--edge", type=float, default=0.03)
    p5.set_defaults(func=cmd_backtest)

    p6 = sub.add_parser("run-all", parents=[common])
    p6.add_argument("--months", type=int, default=3)
    p6.add_argument("--market-limit", type=int, default=500)
    p6.add_argument("--edge", type=float, default=0.03)
    p6.set_defaults(func=cmd_run_all)

    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
