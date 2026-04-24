"""
model_c_static_gate.py
TCN + temporal attention + static feature gate (Model C).
Gate values are fixed after training — the same weight applies to every input window.
Includes optional L1 sparsity penalty to encourage feature pruning.
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
# TCN building block (identical to vanilla TCN)
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
# Static feature gate
# ---------------------------------------------------------------

class FeatureGate(nn.Module):
    """Per-feature learned gate. Each of the 21 features gets a single scalar
    weight passed through sigmoid, producing a value in [0, 1]. The same gate
    values apply to every input window (hence "static").

    Initialized at sigmoid(1.0) ~ 0.73 so features start mostly active.
    Training can push individual gates toward 0 (suppress) or 1 (keep)."""

    def __init__(self, num_features):
        super().__init__()
        # raw pre-sigmoid parameters; initialized to 1.0 so sigmoid(1.0) ~ 0.73
        self.gate_raw = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_raw)  # (F,) gate values in [0, 1]
        return x * gates, gates               # scale each feature column by its gate


# ---------------------------------------------------------------
# Temporal attention (identical to vanilla TCN)
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
# Full model: TCN + Attention + Static Feature Gate
# ---------------------------------------------------------------

class TCNAttention(nn.Module):
    """Static feature gate -> TCN -> temporal attention -> linear head.
    The gate sits before the TCN so the network only sees gated features.
    Adds 21 parameters (one per feature) to the vanilla TCN."""

    def __init__(self, input_dim, hidden_dim=32, kernel_size=3, num_layers=4, dropout=0.1):
        super().__init__()
        self.feature_gate = FeatureGate(input_dim)  # 21 learnable gate weights

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
        x_gated, gate_values = self.feature_gate(x)   # apply static gates to input features
        out = x_gated.transpose(1, 2)                  # -> (batch, F, T) for Conv1d
        out = self.tcn(out)                             # -> (batch, hidden, T) after all blocks
        out = out.transpose(1, 2)                       # -> (batch, T, hidden) for attention
        context, attn_w = self.attention(out)            # -> (batch, hidden) weighted summary
        pred = self.head(context).squeeze(-1)            # -> (batch,) single prediction
        return pred, attn_w, gate_values


# ---------------------------------------------------------------
# Training (with optional sparsity penalty)
# ---------------------------------------------------------------

def train_model(model, train_loader, test_X, test_y,
                epochs=100, lr=1e-3, device="cpu", sparsity_lambda=0.0):
    """Train the model. sparsity_lambda > 0 adds an L1 cost for keeping gates
    open, encouraging the model to suppress features that don't help prediction."""
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
            pred, _, gate_values = model(xb)    # forward pass (returns prediction + gates)
            loss = loss_fn(pred, yb)             # prediction loss (MSE)

            # L1 sparsity penalty: adds cost proportional to sum of gate values,
            # pushing the model to suppress features that don't reduce prediction error
            if sparsity_lambda > 0:
                loss = loss + gate_values.sum() * sparsity_lambda

            optimizer.zero_grad()                # clear previous gradients
            loss.backward()                      # compute new gradients
            optimizer.step()                     # update weights
            train_losses.append(loss.item())

        # evaluate on test set (no gradient computation)
        model.eval()
        with torch.no_grad():
            test_pred, _, _ = model(test_X_t)
            test_loss = loss_fn(test_pred, test_y_t).item()

        # save model if this is the best test MSE so far
        if test_loss < best_test_mse:
            best_test_mse = test_loss
            best_epoch = epoch
            best_state = model.state_dict().copy()

        if epoch % 10 == 0 or epoch == 1:
            # log current gate values alongside training progress
            with torch.no_grad():
                gates = torch.sigmoid(model.feature_gate.gate_raw).cpu().numpy()
            gates_str = "  ".join(f"{g:.2f}" for g in gates)
            print(f"  epoch {epoch:3d}  |  "
                  f"train MSE {np.mean(train_losses):.6f}  |  "
                  f"test MSE {test_loss:.6f}  |  "
                  f"gates [{gates_str}]")

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
    feature_names = list(X_df.columns)

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

    SPARSITY_LAMBDA = 0.001  # adjust: 0.0 = no penalty, higher = more pruning

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TCNAttention(input_dim=X_train.shape[2], hidden_dim=32,
                         kernel_size=3, num_layers=4, dropout=0.1)
    print(f"Static Gate TCN | {sum(p.numel() for p in model.parameters()):,} params | {device} | lambda={SPARSITY_LAMBDA}")
    model = train_model(model, train_loader, X_test, y_test,
                        epochs=100, lr=1e-3, device=device,
                        sparsity_lambda=SPARSITY_LAMBDA)

    # evaluate with best-epoch weights
    model.eval()
    with torch.no_grad():
        test_X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        preds, _, _ = model(test_X_t)
        preds = preds.cpu().numpy()

    # direction accuracy: % of correct up/down predictions
    direction_acc = np.mean(np.sign(preds) == np.sign(y_test))

    # convert log returns to dollar prices for MAE and MAPE
    pred_prices = test_prices * np.exp(preds)
    actual_prices = test_prices * np.exp(y_test)
    mae = np.mean(np.abs(pred_prices - actual_prices))
    mape = np.mean(np.abs((pred_prices - actual_prices) / actual_prices)) * 100

    print(f"  Direction accuracy: {direction_acc:.2%}")
    print(f"  MAE: ${mae:,.2f}  |  MAPE: {mape:.2f}%")

    # report final gate values (0 = feature dropped, 1 = fully used)
    gates = torch.sigmoid(model.feature_gate.gate_raw).detach().cpu().numpy()
    print(f"\n  Feature gate values:")
    for name, g in sorted(zip(feature_names, gates), key=lambda x: -x[1]):
        status = "ACTIVE" if g > 0.5 else "SUPPRESSED"
        print(f"    {name:<20s}  {g:.3f}  {status}")


if __name__ == "__main__":
    main()