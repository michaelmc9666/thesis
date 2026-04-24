# Explainable AI Through Dynamic Feature Gating in Temporal Convolutional Networks for Stock Price Prediction

Master's thesis, University of Colorado Denver, 2026
Author: Michael McGrath
Advisor: Gita Alaghband

## Overview

This repository contains the code for a thesis investigating dynamic feature gating as an intrinsic explainability mechanism for stock price prediction. A small auxiliary network generates per-window gate values between 0 and 1 for each of 21 input features, scaling inputs before they enter the prediction model. These gate values serve as real-time feature importance scores, eliminating the need for expensive post-hoc methods such as SHAP or WindowSHAP.

Four model architectures are compared:

| File | Model | Description | Parameters |
|------|-------|-------------|------------|
| `model_a_lstm.py` | Model A | LSTM baseline | ~7,073 |
| `model_b_vanilla_tcn.py` | Model B | TCN + temporal attention | ~13,442 |
| `model_c_static_gate.py` | Model C | TCN + temporal attention + static feature gate | ~13,463 |
| `model_d_dynamic_gate.py` | Model D | TCN + temporal attention + dynamic feature gate | ~27,659 |

All models share the same data pipeline, hyperparameters (seed 42, 100 epochs, learning rate 1e-3, batch size 128, hidden dimension 32), and evaluation protocol.

## Repository Structure

```
data_pipeline.py           Feature engineering: fetches OHLCV data, computes 21 features, windowing
model_a_lstm.py            Model A: single-layer LSTM baseline
model_b_vanilla_tcn.py     Model B: vanilla TCN with temporal attention
model_c_static_gate.py     Model C: TCN with static (learned, fixed) feature gate
model_d_dynamic_gate.py    Model D: TCN with dynamic per-window feature gate (thesis contribution)
run_all_configs.py         Experiment runner: all tickers x lookbacks x lambdas, outputs results + dashboard
```

## Features

The 21 input features span four categories:

- **Price-derived (3):** log return, log volume, trend (EMA20 − EMA60)
- **Technical (3):** 20-day rolling volatility, RSI, 50-day MA distance
- **Macroeconomic (7):** log VIX, 3-month/1-year/2-year treasury rates, unemployment rate, headline CPI, core CPI
- **Temporal (8):** day-of-year sin/cos, day-of-week sin/cos, four binary season indicators

## Dynamic Gate Architecture

The dynamic gate (Model D) compresses each input window into a 147-dimensional summary vector (per-feature mean + standard deviation + last 5 days), then passes it through a two-layer feedforward network (147 → 84 → 21) with sigmoid output. Each of the 21 output values scales the corresponding feature column before it enters the TCN. The gate bias is initialized to 2.0 so that all features start nearly fully active (sigmoid(2.0) ≈ 0.88).

## Experimental Configuration

- **Tickers:** AAPL, AMZN, JPM, XOM, NEE, GILD, NVDA, LUV
- **Date range:** January 2015 to January 2026
- **Lookback windows:** 300, 600, 900, 1200 trading days
- **Sparsity lambdas:** 0, 0.0001, 0.001
- **Train/test split:** 80/20 chronological

## Usage

### Run all experiments

```bash
python run_all_configs.py
```

This produces a timestamped results text file and an HTML dashboard with per-date gate value lookup.

### Run a single model standalone

Each model file can be run independently for a single ticker:

```bash
python model_d_dynamic_gate.py
```

The default ticker and lookback are configured in `data_pipeline.py`.

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- pandas
- yfinance
- dbnomics (for macroeconomic data)

## Hardware

Experiments were run on an NVIDIA RTX 4080 Laptop GPU. CPU training is possible but significantly slower for longer lookback windows.
