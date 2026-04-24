"""
data_pipeline.py
Feature engineering and windowing pipeline for AAPL daily stock data.
Produces (N, lookback, F) arrays suitable for temporal convolutional networks.
"""

import numpy as np
import pandas as pd
import yfinance as yf

# --- Configuration ---
TICKER = "AAPL"
START  = "2015-01-01"
END    = "2026-01-01"
LOOKBACK = 300       # number of trading days per input window
TRAIN_FRAC = 0.8     # chronological 80/20 train/test split


# ---------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------

def fetch_stock_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    px = yf.download(ticker, start=start, end=end,
                     auto_adjust=False, progress=False)

    if px is None or px.empty:
        raise RuntimeError(
            f"yfinance returned empty data for {ticker} ({start} to {end}).")

    # yfinance sometimes returns MultiIndex columns — flatten them
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)

    # validate required columns exist
    required = {"Adj Close", "Volume", "High", "Low"}
    missing = required - set(px.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    px = px[["Adj Close", "Volume", "High", "Low"]].dropna()
    if px.empty:
        raise RuntimeError("No data remaining after dropping NaNs.")

    # standardize column names and index
    px.index = pd.to_datetime(px.index)
    px.index.name = "date"
    px = px.rename(columns={
        "Adj Close": "adj_close", "Volume": "volume",
        "High": "high", "Low": "low",
    })
    return px


def fetch_vix(start: str, end: str) -> pd.Series:
    """Download VIX (market fear index). Returns None on failure so
    the pipeline can continue without it."""
    try:
        vix = yf.download("^VIX", start=start, end=end,
                          auto_adjust=False, progress=False)
        if vix is None or vix.empty:
            print("  WARNING: VIX download empty. Skipping.")
            return None
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        s = vix["Adj Close"].dropna()
        s.index = pd.to_datetime(s.index)
        s.name = "vix"
        return s
    except Exception as e:
        print(f"  WARNING: VIX download failed ({e}). Skipping.")
        return None


def fetch_macro_features() -> dict:
    """Download macroeconomic features from dbnomics.
    Returns dict of {name: pd.Series}. Skips gracefully on failure."""
    try:
        from dbnomics import fetch_series
    except ImportError:
        print("  WARNING: dbnomics not installed. Skipping macro features.")
        return {}

    macro_series = {}

    # Treasury yields (daily, business days)
    rate_ids = {
        "rate_3m": "FED/H15/RIFLGFCM03_N.B",   # 3-month
        "rate_1y": "FED/H15/RIFLGFCY01_N.B",    # 1-year
        "rate_2y": "FED/H15/RIFLGFCY02_N.B",    # 2-year
    }
    for name, series_id in rate_ids.items():
        try:
            df = fetch_series(series_id)
            df = df[["period", "value"]].copy()
            df["period"] = pd.to_datetime(df["period"])
            df = df.dropna(subset=["value"]).sort_values("period").set_index("period")
            macro_series[name] = df["value"].rename(name)
            print(f"    Downloaded {name} ({len(df)} rows)")
        except Exception as e:
            print(f"    WARNING: {name} failed: {e}")

    # Unemployment rate (monthly)
    try:
        df = fetch_series("BLS/ln/LNS14000000")
        df = df[["period", "value"]].copy()
        df["period"] = pd.to_datetime(df["period"])
        df = df.dropna(subset=["value"]).sort_values("period").set_index("period")
        macro_series["unrate_u3"] = df["value"].rename("unrate_u3")
        print(f"    Downloaded unrate_u3 ({len(df)} rows)")
    except Exception as e:
        print(f"    WARNING: unemployment failed: {e}")

    # CPI levels -> year-over-year inflation rates (monthly)
    # Lagged by 1 month to prevent look-ahead bias (CPI for Jan is released in Feb)
    try:
        cpi_h = fetch_series("BLS/cu/CUSR0000SA0")
        cpi_h = cpi_h[["period", "value"]].copy()
        cpi_h["period"] = pd.to_datetime(cpi_h["period"])
        cpi_h = cpi_h.dropna(subset=["value"]).sort_values("period").set_index("period")

        cpi_c = fetch_series("BLS/cu/CUSR0000SA0L1E")
        cpi_c = cpi_c[["period", "value"]].copy()
        cpi_c["period"] = pd.to_datetime(cpi_c["period"])
        cpi_c = cpi_c.dropna(subset=["value"]).sort_values("period").set_index("period")

        # pct_change(12) = YoY change; shift(1) = lag 1 month for look-ahead prevention
        macro_series["infl_cpi_yoy"] = (cpi_h["value"].pct_change(12) * 100).shift(1).rename("infl_cpi_yoy")
        macro_series["infl_core_yoy"] = (cpi_c["value"].pct_change(12) * 100).shift(1).rename("infl_core_yoy")
        print(f"    Computed inflation features")
    except Exception as e:
        print(f"    WARNING: CPI/inflation failed: {e}")

    return macro_series


# ---------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------

def season_one_hot(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One-hot encode season (winter/spring/summer/fall) from dates."""
    m = dates.month
    season = np.select(
        [m.isin([12, 1, 2]), m.isin([3, 4, 5]),
         m.isin([6, 7, 8]), m.isin([9, 10, 11])],
        ["winter", "spring", "summer", "fall"],
        default="unknown"
    )
    s = pd.Series(season, index=dates, name="season")
    d = pd.get_dummies(s, prefix="season")
    return d.reindex(
        columns=["season_winter", "season_spring", "season_summer", "season_fall"],
        fill_value=0
    )


def cyclical_time_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Encode day-of-year and day-of-week as sin/cos pairs.
    Avoids discontinuity (Dec 31 and Jan 1 are adjacent, not max distance)."""
    doy = dates.dayofyear.to_numpy(dtype=float)
    dow = dates.weekday.to_numpy(dtype=float)
    return pd.DataFrame({
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
    }, index=dates)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, normalized to [0, 1].
    Near 0 = oversold, near 1 = overbought."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)  # avoid div by zero
    rsi = rs / (1 + rs)  # maps to [0, 1] instead of traditional [0, 100]
    rsi.name = "rsi"
    return rsi


# ---------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------

def build_tier0_features(df: pd.DataFrame, start: str = None, end: str = None) -> tuple[pd.DataFrame, pd.Series]:
    """Build all features from raw OHLCV + external sources.
    Returns (features_df, target_series) with NaN rows removed.
    start/end are used for fetching VIX; defaults to module-level START/END."""

    # use module-level defaults if not provided
    _start = start or START
    _end = end or END

    # --- Price-derived features ---
    log_price = np.log(df["adj_close"])
    log_ret = log_price.diff()                           # daily log return
    log_vol = np.log1p(df["volume"])                     # log volume (log1p handles 0)
    ema_fast = log_price.ewm(span=20, adjust=False).mean()
    ema_slow = log_price.ewm(span=60, adjust=False).mean()
    trend = ema_fast - ema_slow                          # short vs long-term momentum

    # --- Technical indicators ---
    volatility_20d = log_ret.rolling(window=20).std()    # 20-day rolling volatility
    volatility_20d.name = "volatility_20d"
    rsi = compute_rsi(df["adj_close"], period=14)        # 14-day RSI
    ma_50 = df["adj_close"].rolling(window=50).mean()
    ma_distance = ((df["adj_close"] - ma_50) / ma_50)    # % distance from 50-day MA
    ma_distance.name = "ma50_distance"

    # --- Time/seasonality features ---
    time_cyc = cyclical_time_features(df.index)
    seasons = season_one_hot(df.index)

    # --- External data: VIX ---
    vix_series = fetch_vix(_start, _end)

    # --- External data: macroeconomic indicators ---
    print("  Fetching macro data from dbnomics ...")
    macro_dict = fetch_macro_features()

    # --- Combine all features into one DataFrame ---
    feat = pd.DataFrame({
        "log_ret": log_ret,
        "log_vol": log_vol,
        "trend": trend,
        "volatility_20d": volatility_20d,
        "rsi": rsi,
        "ma50_distance": ma_distance,
    }, index=df.index).join([time_cyc, seasons], how="left")

    # VIX: log-transformed, forward-filled to cover non-trading days
    if vix_series is not None:
        log_vix = np.log(vix_series).rename("log_vix")
        feat = feat.join(log_vix, how="left")
        feat["log_vix"] = feat["log_vix"].ffill()

    # Macro: forward-filled from monthly/business-daily to trading-daily
    for name, series in macro_dict.items():
        feat = feat.join(series, how="left")
        feat[name] = feat[name].ffill()

    # --- Target: next-day log return ---
    y = feat["log_ret"].shift(-1).rename("y_next_log_ret")

    # Drop any rows with NaN (from rolling windows, diffs, or missing data)
    clean = feat.join(y, how="left").dropna()
    y_clean = clean["y_next_log_ret"]
    X_clean = clean.drop(columns=["y_next_log_ret"])

    return X_clean, y_clean


# ---------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------

def standardize_train_only(X: pd.DataFrame, train_end_idx: int) -> pd.DataFrame:
    """Z-score standardization using only training set statistics.
    Prevents data leakage from test set into training."""
    X_train = X.iloc[:train_end_idx]
    mu = X_train.mean()
    sd = X_train.std().replace(0, 1.0)  # avoid div by zero for constant columns
    return (X - mu) / sd


# ---------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------

def make_windows(X: pd.DataFrame, y: pd.Series, lookback: int, train_end_idx: int):
    """Create sliding windows of shape (N, lookback, F).
    Each window's target is the value at the last timestep.
    Train/test split is chronological (no future data in training)."""
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")

    T = len(X)
    if T < lookback:
        raise ValueError(f"Not enough rows ({T}) for lookback={lookback}")

    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    N = T - lookback + 1   # total number of windows

    # Pre-allocate arrays
    Xw = np.zeros((N, lookback, Xv.shape[1]), dtype=float)
    yw = np.zeros((N,), dtype=float)

    # Slide window across the time series
    for i in range(N):
        Xw[i] = Xv[i: i + lookback]       # window of lookback days
        yw[i] = yv[i + lookback - 1]       # target at end of window

    # Split by window end position (preserves chronological order)
    end_positions = np.arange(lookback - 1, T)
    train_mask = end_positions < train_end_idx

    X_train, y_train = Xw[train_mask], yw[train_mask]
    X_test, y_test = Xw[~train_mask], yw[~train_mask]

    if X_train.shape[0] == 0:
        raise ValueError("No training windows created.")
    if X_test.shape[0] == 0:
        raise ValueError("No test windows created.")

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------
# Main (standalone test)
# ---------------------------------------------------------------

def main():
    df = fetch_stock_ohlcv(TICKER, START, END)
    print(f"Raw OHLCV rows: {len(df)} | {df.index.min().date()} -> {df.index.max().date()}")

    X_df, y = build_tier0_features(df)
    print(f"After feature eng: {len(X_df)} rows | {X_df.index.min().date()} -> {X_df.index.max().date()}")

    if len(X_df) < LOOKBACK:
        raise ValueError(f"Not enough rows: {len(X_df)} < LOOKBACK={LOOKBACK}")

    train_end_idx = int(TRAIN_FRAC * len(X_df))
    X_scaled = standardize_train_only(X_df, train_end_idx)

    X_train, y_train, X_test, y_test = make_windows(
        X_scaled, y, lookback=LOOKBACK, train_end_idx=train_end_idx)

    print(f"\nFeatures ({len(X_df.columns)}): {list(X_df.columns)}")
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")


if __name__ == "__main__":
    main()