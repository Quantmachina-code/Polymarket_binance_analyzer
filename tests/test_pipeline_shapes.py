import pandas as pd

from pipeline import binance_quality_report, joined_quality_report


def test_binance_quality_report_basic():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
            "symbol": ["BTCUSDC", "BTCUSDC", "BTCUSDC"],
            "close": [1, 2, 3],
            "volume": [1.0, 2.0, 1.5],
        }
    )
    report = binance_quality_report(df)
    assert len(report) == 1
    assert "missing_minutes_gaps" in report.columns


def test_joined_quality_report_basic():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC"),
            "target_up_5m": [0, 1, 0, 1],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    report = joined_quality_report(df)
    assert int(report.loc[0, "rows"]) == 4
