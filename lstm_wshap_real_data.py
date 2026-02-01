# demo_windowshap_finance_long_series.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from dbnomics import fetch_series
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd


from windowshap import StationaryWindowSHAP


def fetch_macro(series_id: str, value_col: str ="value") -> pd.Series:
    """
    Returns a 1D (1 value per timestamp) panda data series

    - 1D series: 1 column of numbers over time (time is index)
    - each macro indicator is a separate 1D series
    """
    df = fetch_series(series_id)                        # dbnomics returns a dataframe
    df = df[["period", value_col]].copy()               # keep only date + value
    df["period"] = pd.to_datetime(df["period"])         # convert string -> datetime
    df = df.dropna(subset=[value_col])                  # drop missing values
    df = df.sort_values("period").set_index("period")   # sort, then set datetime as index
    return df[value_col].rename(series_id)              # return as series named by id


def make_real_series(ticker, start, end, price_col):
    """
    Downloads real market data and returns:
      - series: (T, 2) numpy array with [price, volume]
      - prices: (T,) numpy array
      - vol:    (T,) numpy array
    This matches the shape your synthetic code expects.
    """

    # download OHLCV from Yahoo Finance
    px = yf.download(
        ticker,                 # which stock
        start=start,             # start date (inclusive)
        end=end,                 # end date (exclusive)
        auto_adjust=False,       # keep Adj Close column
        progress=False
    )

    # keep only what we need, and drop any missing rows
    px = px[[price_col, "Volume"]].dropna()

    # extract 1D arrays (T,), squeeze() forces arrays to be 1D
    prices = px[price_col].to_numpy(dtype=float).squeeze()
    vol = px["Volume"].to_numpy(dtype=float).squeeze()


    # stack into (T, 2) where column0=price, column1=volume
    series = np.stack((prices, vol), axis=-1)

    return series, prices, vol



def build_window_dataset(series_norm, lookback=300):
    T_long, F = series_norm.shape   #t_long = # of rows, and F=number of columns
    N = T_long - lookback           # number of rows (excludes x rows, where x is the lookback length)

    X = np.stack([series_norm[i:i + lookback] for i in range(N)], axis=0)
    y = X[:, -1, 0]    # last (normalized) price in each window
    return X, y


def build_lstm_model(lookback, num_features):
    # simple sequence-to-one model:
    # input shape = (lookback timesteps, num_features)
    model = models.Sequential([
        layers.Input(shape=(lookback, num_features)),  # expects windows shaped like X[i]
        layers.LSTM(32),                               # reads full window -> 32-d hidden state
        layers.Dense(1)                                # maps hidden state -> single scalar prediction
    ])
    # configure how the model will be trained (optimizer + loss)
    model.compile(optimizer="adam", loss="mse")
    return model


# ---------- WindowSHAP wrapper ----------

def run_windowshap(model, X_train, X_test,
                   window_len=10, num_bg=40, num_test_shap=3):
    """
    Run StationaryWindowSHAP on a few test windows.
    Returns ts_phi so plotting is separate.
    """
    # choose up to num_bg background windows spread across training set
    num_bg = min(num_bg, X_train.shape[0])
    bg_indices = np.linspace(0, X_train.shape[0] - 1, num_bg, dtype=int)
    B_ts = X_train[bg_indices]  # shape: (num_bg, lookback, 2)

    # pick the last few test windows to explain (most recent data)
    num_test_shap = min(num_test_shap, X_test.shape[0])
    test_ts_shap = X_test[-num_test_shap:]  # shape: (num_test_shap, lookback, 2)

    # create WindowSHAP explainer that talks to our trained Keras model
    ws_explainer = StationaryWindowSHAP(
        model=model,
        window_len=window_len,   # internal time-window size for SHAP (e.g., 10 steps)
        B_ts=B_ts,
        test_ts=test_ts_shap,
        model_type='lstm'        # tells the wrapper that the model takes only time-series input
    )

    # ts_phi shape: (num_test_shap, lookback, num_features)
    ts_phi = ws_explainer.shap_values()
    print("WindowSHAP ts_phi shape:", ts_phi.shape)
    print("ts_phi min/max:", ts_phi.min(), ts_phi.max())
    return ts_phi

# ---------- plotting helpers ----------

def plot_price_volume(prices, vol, max_steps=500):
    # make a simple timeline 0..max_steps-1 for plotting
    t = np.arange(max_steps)
    # 2 rows, 1 column of subplots: top = price, bottom = volume
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axs[0].plot(t, prices[:max_steps])
    axs[0].set_ylabel("Price")
    axs[0].set_title("Real Price and Volume (first 500 steps)")
    axs[1].plot(t, vol[:max_steps])
    axs[1].set_ylabel("Volume")
    axs[1].set_xlabel("Time step")
    plt.tight_layout()
    plt.show()


def plot_price_predictions(true_last_prices, pred_last_prices, max_plot=200):
    # only plot up to max_plot windows to keep the chart readable
    max_plot = min(max_plot, len(true_last_prices))
    plt.figure(figsize=(10, 4))
    # blue = ground-truth, orange = model prediction
    plt.plot(true_last_prices[:max_plot], label="true last price in window", alpha=0.7)
    plt.plot(pred_last_prices[:max_plot], label="pred last price in window", alpha=0.7)
    plt.title("Target: last price in each window (first 200 test windows)")
    plt.xlabel("Window index (time order)")  # x-axis is just index in the test set
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_windowshap(ts_phi, window_index=0):
    # ts_phi is (num_test_shap, lookback, 2); we take one sample and transpose to (2, lookback)
    plt.figure(figsize=(8, 4))
    plt.imshow(ts_phi[window_index].T, aspect="auto", interpolation="nearest")
    plt.colorbar(label="SHAP value (impact on last price)")
    # row 0 = price feature, row 1 = volume feature
    plt.yticks([0, 1], ["price (norm)", "volume (norm)"])
    plt.xlabel("time step within window")
    plt.ylabel("feature")
    plt.title("WindowSHAP explanation for one test window")
    plt.tight_layout()
    plt.show()

# ---------- main pipeline ----------

def main():
    print(">>> starting main()")

    # 1) make long REAL series (price + volume)
    series, prices, vol = make_real_series(
        ticker="AAPL",
        start="2015-01-01",
        end="2026-01-01",
        price_col="Adj Close"
    )

    # this is a dbnomics test before I actually use it
    #------------------------------------------------------------
    series_id_1 = "FED/H15/RIFLGFCM03_N.B"    # provider/dataset/series_code
    # us treasury constant maturity (nominal) yield
    # 3 month maturity, business-day freq.

    series_id_2 = "FED/H15/RIFLGFCY01_N.B"  # provider/dataset/series_code
    # us treasury constant maturity (nominal) yield
    # 1 year maturity, business-day freq.

    series_id_3 = "FED/H15/RIFLGFCY02_N.B"  # provider/dataset/series_code
    # us treasury constant maturity (nominal) yield
    # 2 year maturity, business-day freq.


    tcmnom_3m = fetch_macro(series_id_1)
    tcmnom_1y = fetch_macro(series_id_2)
    tcmnom_2y = fetch_macro(series_id_3)

    print("\ntcmnom 3 months\n")
    print(tcmnom_3m.name)       # confirms which series was pulled
    print(tcmnom_3m.index[:3])  # dates
    print(tcmnom_3m.head())     # first values

    print("\ntcmnom 1 year\n")
    print(tcmnom_1y.name)  # confirms which series was pulled
    print(tcmnom_1y.index[:3])  # dates
    print(tcmnom_1y.head())  # first values

    print("\ntcmnom 2 years\n")
    print(tcmnom_2y.name)  # confirms which series was pulled
    print(tcmnom_2y.index[:3])  # dates
    print(tcmnom_2y.head())  # first values

    # ------------------------------------------------------------


    # normalize inputs (convert raw price/volume to roughly N(0,1))
    price_mean, price_std = prices.mean(), prices.std()
    vol_mean, vol_std = vol.mean(), vol.std()

    prices_norm = (prices - price_mean) / price_std
    vol_norm = (vol - vol_mean) / vol_std
    # stack normalized price + volume back into 2D array (time, features)
    series_norm = np.stack([prices_norm, vol_norm], axis=-1)  # (T_long, 2)

    # plot raw (unnormalized) price/volume just to see the synthetic data
    plot_price_volume(prices, vol, max_steps=500)

    # 2) window dataset (normalized) with 300-step windows
    lookback = 300
    X, y = build_window_dataset(series_norm, lookback=lookback)
    print("X shape:", X.shape)   # (N, 300, 2) = (num_windows, timesteps, features)
    print("y shape:", y.shape)   # (N,) = last normalized price per window

    # 3) train/test split (respect time order: earliest -> train, latest -> test)
    N = X.shape[0]
    split_idx = int(0.8 * N)               # 80% train, 20% test
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 4) build + train LSTM on (window -> last price) mapping
    model = build_lstm_model(lookback=lookback, num_features=X.shape[2])
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

    # 5) predictions (unnormalized last prices)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print("Test MSE loss (on normalized prices):", test_loss)

    # model outputs normalized predictions; flatten to 1D (N_test,)
    y_pred = model.predict(X_test, verbose=0)[:, 0]

    # convert normalized targets + predictions back to real price scale
    true_last_prices = y_test * price_std + price_mean
    pred_last_prices = y_pred * price_std + price_mean

    # plot true vs predicted last prices for a slice of the test windows
    plot_price_predictions(true_last_prices, pred_last_prices, max_plot=200)

    # 6) WindowSHAP: explain which timesteps/features matter for a few test windows
    ts_phi = run_windowshap(model, X_train, X_test, window_len=10)
    plot_windowshap(ts_phi, window_index=0)

    print(">>> finished main()")


if __name__ == "__main__":
    main()
