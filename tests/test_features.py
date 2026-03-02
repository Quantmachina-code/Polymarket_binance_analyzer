import pandas as pd

from pipeline import add_technical_features


def test_add_technical_features_creates_expected_columns():
    df = pd.DataFrame({"close": [100 + i for i in range(30)]})
    out = add_technical_features(df, "close", "btc")
    for col in ["btc_ret_1", "btc_ret_5", "btc_vol_15", "btc_ma_cross"]:
        assert col in out.columns
