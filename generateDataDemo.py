import yfinance as yf
import pandas as pd
import numpy as np

# step 1: download raw stock tables

#--- chose what to download ---
ticker = "AAPL"
start = "2015-01-01"
end = "2026-01-01"

#--- download daily OHLCV data ---
px = yf.download(
    ticker,
    start=start,
    end=end,
    auto_adjust=False,
    progress=False
)


prices = px["Adj Close"].to_numpy()         # make 1D array of adjusted close prices
vol = px["Volume"].to_numpy()               # 1D array of volumes

# built (T,2) series: [price, volume], axis=-1
series = np.stack((prices, vol), axis=-1)

# compute price/vol mean and standard deviation for normalization
price_mean, price_std = prices.mean(), prices.std()
vol_mean, vol_std = vol.mean(), vol.std()

# normalizes price and vol, and finally the series
prices_norm = (prices - price_mean) / price_std
vol_norm = (vol - vol_mean) / vol_std
series_norm = np.stack((prices_norm, vol_norm), axis=-1)
