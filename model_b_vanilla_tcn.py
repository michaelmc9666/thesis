"""
model_b_vanilla_tcn.py
Vanilla TCN + temporal attention (Model B). No feature gating.
Serves as the baseline TCN against which gated variants are compared.
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
# TCN building block
# ---------------------------------------------------------------

class CausalConv1dBlock(nn.Module):
    """Single TCN layer: dilated causal conv -> batch norm -> ReLU -> dropout,
    with a residual skip connection for stable gradient flow."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation  # left-padding for causal alignment

        # dilated 1D convolution (no built-in padding; we pad manually for causality)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        self.bn = nn.BatchNorm1d(out_ch)         # normalize activations per channel
        self.relu = nn.ReLU()                     # zero out negatives (nonlinearity)
        self.dropout = nn.Dropout(dropout)         # randomly zero 10% of values (regularization)

        # residual path: 1x1 conv to match dimensions if in_ch != out_ch, else identity
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)                          # save input for skip connection
        out = nn.functional.pad(x, (self.pad, 0))        # causal padding (zeros on left only)
        out = self.conv(out)                              # apply dilated convolution
        out = self.bn(out)                                # batch normalization
        out = self.relu(out)                              # activation
        out = self.dropout(out)                           # dropout (training only)
        return out + residual                             # add skip connection


# ---------------------------------------------------------------
# Temporal attention
# ---------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Additive attention over the time axis.
    Collapses (batch, T, hidden) -> (batch, hidden) via learned weighted sum."""

    def __init__(self, hidden_dim):
        super().__init__()
        # two-layer score network: hidden -> hidden -> 1 score per timestep
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h):
        scores = self.score_net(h)                 # (batch, T, 1) — raw importance scores
        weights = torch.softmax(scores, dim=1)     # (batch, T, 1) — normalized to sum to 1
        context = (weights * h).sum(dim=1)         # (batch, hidden) — weighted average
        return context, weights.squeeze(-1)


# ---------------------------------------------------------------
# Full model: TCN + Attention (no gating)
# ---------------------------------------------------------------

class TCNAttention(nn.Module):
    """TCN encoder with temporal attention pooling and linear prediction head.
    All 21 features pass through at full strength (no gating)."""

    def __init__(self, input_dim, hidden_dim=32, kernel_size=3, num_layers=4, dropout=0.1):
        super().__init__()

        # stack TCN blocks with exponentially increasing dilation (1, 2, 4, 8)
        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim  # first block: F->32, rest: 32->32
            layers.append(CausalConv1dBlock(in_ch, hidden_dim, kernel_size,
                                            dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*layers)

        self.attention = TemporalAttention(hidden_dim)  # collapse T timesteps -> 1 vector
        self.head = nn.Linear(hidden_dim, 1)            # map hidden vector -> 1 prediction

    def forward(self, x):
        # x: (batch, lookback, F)
        out = x.transpose(1, 2)                # -> (batch, F, T) for Conv1d
        out = self.tcn(out)                     # -> (batch, hidden, T) after all blocks
        out = out.transpose(1, 2)               # -> (batch, T, hidden) for attention
        context, attn_w = self.attention(out)    # -> (batch, hidden) weighted summary
        pred = self.head(context).squeeze(-1)    # -> (batch,) single prediction per sample
        return pred, attn_w


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
            pred, _ = model(xb)              # forward pass
            loss = loss_fn(pred, yb)          # compute MSE
            optimizer.zero_grad()             # clear previous gradients
            loss.backward()                   # compute new gradients
            optimizer.step()                  # update weights
            train_losses.append(loss.item())

        # evaluate on test set (no gradient computation)
        model.eval()
        with torch.no_grad():
            test_pred, _ = model(test_X_t)
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
    model = TCNAttention(input_dim=X_train.shape[2], hidden_dim=32,
                         kernel_size=3, num_layers=4, dropout=0.1)
    print(f"Vanilla TCN + Attention | {sum(p.numel() for p in model.parameters()):,} params | {device}")
    model = train_model(model, train_loader, X_test, y_test, epochs=100, lr=1e-3, device=device)

    # evaluate with best-epoch weights
    model.eval()
    with torch.no_grad():
        test_X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        preds, _ = model(test_X_t)
        preds = preds.cpu().numpy()

    # direction accuracy and price-level metrics
    direction_acc = np.mean(np.sign(preds) == np.sign(y_test))
    pred_prices = test_prices * np.exp(preds)
    actual_prices = test_prices * np.exp(y_test)
    mae = np.mean(np.abs(pred_prices - actual_prices))
    mape = np.mean(np.abs((pred_prices - actual_prices) / actual_prices)) * 100

    print(f"  Direction accuracy: {direction_acc:.2%}")
    print(f"  MAE: ${mae:,.2f}  |  MAPE: {mape:.2f}%")


if __name__ == "__main__":
    main()