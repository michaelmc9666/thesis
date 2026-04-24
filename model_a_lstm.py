"""
model_a_lstm.py
Single-layer LSTM baseline (Model A) for comparison against TCN variants.
Uses the same data pipeline, features, windowing, and train/test split
so results are directly comparable.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(42)
np.random.seed(42)

from data_pipeline import (
    fetch_stock_ohlcv, build_tier0_features, standardize_train_only,
    make_windows, TICKER, START, END, LOOKBACK, TRAIN_FRAC,
)


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------

class LSTMBaseline(nn.Module):
    """Single-layer LSTM with a linear head for regression.
    Takes (batch, lookback, F) input and outputs (batch,) predictions.
    Uses hidden_dim=32 to match the TCN's capacity for fair comparison."""

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)  # maps 32-dim hidden state to 1 scalar prediction

    def forward(self, x):
        # x: (batch, lookback, F)
        _, (h_n, _) = self.lstm(x)    # h_n: (1, batch, hidden_dim) — final hidden state
        pred = self.head(h_n.squeeze(0)).squeeze(-1)  # (batch,) — single prediction per sample
        return pred


# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------

def train_model(model, train_loader, test_X, test_y,
                epochs=100, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # pre-convert test data to tensors (stays on device for all epochs)
    test_X_t = torch.tensor(test_X, dtype=torch.float32, device=device)
    test_y_t = torch.tensor(test_y, dtype=torch.float32, device=device)

    # track best model state to avoid using an overfit later epoch
    best_test_mse = float('inf')
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)              # forward pass
            loss = loss_fn(pred, yb)      # compute MSE
            optimizer.zero_grad()         # clear previous gradients
            loss.backward()               # compute new gradients
            optimizer.step()              # update weights
            train_losses.append(loss.item())

        # evaluate on test set (no gradient computation)
        model.eval()
        with torch.no_grad():
            test_pred = model(test_X_t)
            test_loss = loss_fn(test_pred, test_y_t).item()

        # save model if this is the best test MSE so far
        if test_loss < best_test_mse:
            best_test_mse = test_loss
            best_epoch = epoch
            best_state = model.state_dict().copy()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  |  train MSE {np.mean(train_losses):.6f}  |  test MSE {test_loss:.6f}")

    print(f"  Best test MSE: {best_test_mse:.6f} at epoch {best_epoch}")
    model.load_state_dict(best_state)  # restore best epoch weights
    return model


# ---------------------------------------------------------------
# Main (standalone run)
# ---------------------------------------------------------------

def main():
    # load and preprocess data
    df = fetch_stock_ohlcv(TICKER, START, END)
    X_df, y = build_tier0_features(df)

    train_end_idx = int(TRAIN_FRAC * len(X_df))
    X_scaled = standardize_train_only(X_df, train_end_idx)

    # save dates and prices before converting to numpy arrays
    all_dates = X_df.index
    all_prices = df["adj_close"].reindex(all_dates)

    X_train, y_train, X_test, y_test = make_windows(
        X_scaled, y, lookback=LOOKBACK, train_end_idx=train_end_idx)

    # map each test window to the calendar date and closing price of its last day
    T = len(X_df)
    end_positions = np.arange(LOOKBACK - 1, T)
    train_mask = end_positions < train_end_idx
    test_end_positions = end_positions[~train_mask]
    test_dates = [all_dates[pos] for pos in test_end_positions]
    test_prices = np.array([all_prices.iloc[pos] for pos in test_end_positions])

    # build and train model
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMBaseline(input_dim=X_train.shape[2], hidden_dim=32)
    print(f"LSTM Baseline | {sum(p.numel() for p in model.parameters()):,} params | {device}")
    model = train_model(model, train_loader, X_test, y_test, epochs=100, lr=1e-3, device=device)

    # evaluate with best-epoch weights
    model.eval()
    with torch.no_grad():
        test_X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        preds = model(test_X_t).cpu().numpy()

    # direction accuracy: % of correct up/down predictions
    direction_acc = np.mean(np.sign(preds) == np.sign(y_test))

    # convert log returns to dollar prices for MAE and MAPE
    # predicted price = today's close * exp(predicted log return)
    # actual price    = today's close * exp(actual log return)
    pred_prices = test_prices * np.exp(preds)
    actual_prices = test_prices * np.exp(y_test)
    mae = np.mean(np.abs(pred_prices - actual_prices))
    mape = np.mean(np.abs((pred_prices - actual_prices) / actual_prices)) * 100

    print(f"  Direction accuracy: {direction_acc:.2%}")
    print(f"  MAE: ${mae:,.2f}  |  MAPE: {mape:.2f}%")


if __name__ == "__main__":
    main()