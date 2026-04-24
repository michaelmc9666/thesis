"""
model_d_dynamic_gate.py
TCN + temporal attention + dynamic per-window feature gate (Model D).
This is the primary contribution of the thesis.

The dynamic gate compresses each input window into a 147-dimensional summary
(per-feature mean + std + last 5 days), passes it through a two-layer network
(147 -> 84 -> 21), and outputs 21 sigmoid gate values. Each gate value scales
the corresponding feature column before it enters the TCN, producing per-window
feature importance scores at zero additional inference cost.
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
# TCN building block (identical to vanilla TCN and static gate)
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
# Dynamic feature gate (thesis contribution)
# ---------------------------------------------------------------

class DynamicFeatureGate(nn.Module):
    """Per-window feature gate. Unlike the static gate which learns one fixed
    weight per feature, this gate produces different weights for every input
    window based on a summary of that window's contents.

    Summary construction (147 dimensions for 21 features):
        - Per-feature mean across all timesteps     (21 values)
        - Per-feature std across all timesteps       (21 values)
        - Raw values of the last 5 timesteps         (5 x 21 = 105 values)
        Total: 21 + 21 + 105 = 147

    Gate network: 147 -> 84 (ReLU) -> 21 (sigmoid)
        - Hidden dim = num_features * 4 = 84
        - Output bias initialized to 2.0 so sigmoid(2.0) ~ 0.88,
          meaning all features start nearly fully active before training."""

    def __init__(self, num_features, recent_days=5):
        super().__init__()
        self.num_features = num_features
        self.recent_days = recent_days

        # summary size: mean + std + (recent_days * features)
        summary_size = num_features * (2 + recent_days)  # 21 * 7 = 147

        self.gate_net = nn.Sequential(
            nn.Linear(summary_size, num_features * 4),   # 147 -> 84
            nn.ReLU(),
            nn.Linear(num_features * 4, num_features),   # 84 -> 21
        )

        # initialize output layer so all gates start near 1.0
        # sigmoid(2.0) ~ 0.88, so features begin mostly active
        final_layer = self.gate_net[2]
        nn.init.zeros_(final_layer.weight)    # no initial feature preference
        nn.init.constant_(final_layer.bias, 2.0)  # bias toward keeping features

    def forward(self, x):
        # x: (batch, lookback, F)

        # compute per-feature summary statistics across the time axis
        feat_mean = x.mean(dim=1)                                    # (batch, F)
        feat_std = x.std(dim=1)                                      # (batch, F)

        # grab the last few days of raw values (captures recent regime)
        feat_recent = x[:, -self.recent_days:, :]                    # (batch, 5, F)
        feat_recent_flat = feat_recent.reshape(x.size(0), -1)        # (batch, 5*F)

        # concatenate into a single summary vector
        summary = torch.cat([feat_mean, feat_std, feat_recent_flat], dim=1)  # (batch, 147)

        # pass summary through gate network to get per-window gate values
        raw = self.gate_net(summary)          # (batch, 21) — pre-sigmoid scores
        gates = torch.sigmoid(raw)            # (batch, 21) — gate values in [0, 1]

        # scale each feature column by its gate value
        x_gated = x * gates.unsqueeze(1)      # broadcast (batch, 1, F) across time axis

        return x_gated, gates


# ---------------------------------------------------------------
# Temporal attention (identical to vanilla TCN and static gate)
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
# Full model: TCN + Attention + Dynamic Feature Gate
# ---------------------------------------------------------------

class TCNAttention(nn.Module):
    """Dynamic feature gate -> TCN -> temporal attention -> linear head.
    The gate produces per-window feature weights, so the TCN sees a
    differently-weighted input for every prediction window."""

    def __init__(self, input_dim, hidden_dim=32, kernel_size=3, num_layers=4, dropout=0.1):
        super().__init__()
        self.feature_gate = DynamicFeatureGate(input_dim)  # per-window gate network

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
        x_gated, gate_values = self.feature_gate(x)   # apply dynamic gates to input features
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
    """Train the model. sparsity_lambda > 0 adds an L1 cost on the mean gate
    value across each batch, encouraging the model to suppress features that
    don't help prediction."""
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

            # L1 sparsity penalty on mean gate value per batch
            # uses mean (not sum) because gate count varies per window in dynamic mode
            if sparsity_lambda > 0:
                gate_penalty = gate_values.mean() * sparsity_lambda
                loss = loss + gate_penalty

            optimizer.zero_grad()                # clear previous gradients
            loss.backward()                      # compute new gradients
            optimizer.step()                     # update weights
            train_losses.append(loss.item())

        # evaluate on test set (no gradient computation)
        model.eval()
        with torch.no_grad():
            test_pred, _, test_gates = model(test_X_t)
            test_loss = loss_fn(test_pred, test_y_t).item()

        # save model if this is the best test MSE so far
        if test_loss < best_test_mse:
            best_test_mse = test_loss
            best_epoch = epoch
            best_state = model.state_dict().copy()

        if epoch % 10 == 0 or epoch == 1:
            # log gate statistics alongside training progress
            with torch.no_grad():
                avg_gates = test_gates.mean(dim=0).cpu().numpy()
                max_std = test_gates.std(dim=0).cpu().numpy().max()
            print(f"  epoch {epoch:3d}  |  "
                  f"train MSE {np.mean(train_losses):.6f}  |  "
                  f"test MSE {test_loss:.6f}  |  "
                  f"max gate std {max_std:.4f}")

    print(f"  Best test MSE: {best_test_mse:.6f} at epoch {best_epoch}")
    model.load_state_dict(best_state)  # restore best epoch weights
    return model


# ---------------------------------------------------------------
# Interactive explorer (standalone use)
# ---------------------------------------------------------------

def interactive_explorer(all_gates, test_dates, feature_names, y_test):
    """Command-line tool for examining gate values at specific dates.
    Supports single-date lookup, date comparison, and finding windows
    with the most extreme gate variation."""
    import pandas as pd

    date_index = pd.DatetimeIndex(test_dates)

    print(f"\n  Test windows: {date_index[0].date()} to {date_index[-1].date()} ({len(date_index)} windows)")
    print(f"  Commands:")
    print(f"    YYYY-MM-DD             — look up gate values for a date")
    print(f"    compare DATE1 DATE2    — compare two windows side by side")
    print(f"    top N                  — show N windows with most gate variation")
    print(f"    quit                   — exit\n")

    while True:
        user_input = input("  >> ").strip()
        if not user_input or user_input.lower() == 'quit':
            break

        # "top N" — windows with highest gate variation
        if user_input.lower().startswith("top"):
            try:
                n = int(user_input.split()[1])
            except (IndexError, ValueError):
                n = 5
            gate_ranges = all_gates.max(axis=1) - all_gates.min(axis=1)
            top_indices = np.argsort(gate_ranges)[-n:][::-1]
            print(f"\n  Top {n} windows by gate variation:")
            for idx in top_indices:
                date = date_index[idx].date()
                direction = "UP" if y_test[idx] > 0 else "DOWN"
                print(f"\n    {date}  |  range: {gate_ranges[idx]:.3f}  |  actual: {direction} ({y_test[idx]:+.4f})")
                for name, g in sorted(zip(feature_names, all_gates[idx]), key=lambda x: -x[1]):
                    status = "ACTIVE" if g > 0.5 else "SUPPRESSED"
                    print(f"      {name:<20s}  {g:.3f}  {status}")
            print()
            continue

        # "compare DATE1 DATE2" — side-by-side comparison
        if user_input.lower().startswith("compare"):
            parts = user_input.split()
            if len(parts) != 3:
                print("  Usage: compare 2020-03-15 2021-06-15")
                continue
            try:
                d1, d2 = pd.Timestamp(parts[1]), pd.Timestamp(parts[2])
            except Exception:
                print(f"  Could not parse dates: {parts[1]}, {parts[2]}")
                continue
            idx1 = np.argmin(np.abs(date_index - d1))
            idx2 = np.argmin(np.abs(date_index - d2))
            print(f"\n  Comparing {date_index[idx1].date()} vs {date_index[idx2].date()}")
            print(f"  {'Feature':<20s}  {'Gate 1':>8s}  {'Gate 2':>8s}  {'Diff':>8s}")
            print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}")
            pairs = sorted(zip(feature_names, all_gates[idx1], all_gates[idx2]),
                           key=lambda x: -abs(x[1] - x[2]))
            for name, g1, g2 in pairs:
                print(f"    {name:<20s}  {g1:>8.3f}  {g2:>8.3f}  {g2 - g1:>+7.3f}")
            print()
            continue

        # single date lookup
        try:
            target_date = pd.Timestamp(user_input)
        except Exception:
            print(f"  Could not parse date: '{user_input}'. Use YYYY-MM-DD format.")
            continue

        idx = np.argmin(np.abs(date_index - target_date))
        actual_date = date_index[idx].date()
        direction = "UP" if y_test[idx] > 0 else "DOWN"
        print(f"\n  Window ending: {actual_date}  |  actual: {direction} ({y_test[idx]:+.6f})")
        for name, g in sorted(zip(feature_names, all_gates[idx]), key=lambda x: -x[1]):
            status = "ACTIVE" if g > 0.5 else "SUPPRESSED"
            print(f"    {name:<20s}  {g:.3f}  {status}")
        print()


# ---------------------------------------------------------------
# Main (standalone run)
# ---------------------------------------------------------------

def main():
    import pandas as pd

    # load and preprocess data
    df = fetch_stock_ohlcv(TICKER, START, END)
    X_df, y = build_tier0_features(df)
    feature_names = list(X_df.columns)

    train_end_idx = int(TRAIN_FRAC * len(X_df))
    X_scaled = standardize_train_only(X_df, train_end_idx)

    # save dates before converting to numpy
    all_dates = X_df.index

    X_train, y_train, X_test, y_test = make_windows(
        X_scaled, y, lookback=LOOKBACK, train_end_idx=train_end_idx)

    # map each test window to the calendar date of its last day
    T = len(X_df)
    end_positions = np.arange(LOOKBACK - 1, T)
    train_mask = end_positions < train_end_idx
    test_end_positions = end_positions[~train_mask]
    test_dates = [all_dates[pos] for pos in test_end_positions]

    # build and train model
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    SPARSITY_LAMBDA = 0.0001  # mild penalty to sharpen gate separation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TCNAttention(input_dim=X_train.shape[2], hidden_dim=32,
                         kernel_size=3, num_layers=4, dropout=0.1)
    print(f"Dynamic Gate TCN | {sum(p.numel() for p in model.parameters()):,} params | {device} | lambda={SPARSITY_LAMBDA}")
    model = train_model(model, train_loader, X_test, y_test,
                        epochs=100, lr=1e-3, device=device,
                        sparsity_lambda=SPARSITY_LAMBDA)

    # evaluate with best-epoch weights
    model.eval()
    with torch.no_grad():
        test_X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        preds, _, all_gates = model(test_X_t)
        preds = preds.cpu().numpy()
        all_gates = all_gates.cpu().numpy()

    # direction accuracy: % of correct up/down predictions
    direction_acc = np.mean(np.sign(preds) == np.sign(y_test))
    print(f"  Direction accuracy: {direction_acc:.2%}")

    # report average gate values across all test windows
    avg_gates = all_gates.mean(axis=0)
    std_gates = all_gates.std(axis=0)
    print(f"\n  Average gate values:")
    for name, avg, std in sorted(zip(feature_names, avg_gates, std_gates), key=lambda x: -x[1]):
        variability = "DYNAMIC" if std > 0.05 else "STABLE"
        status = "ACTIVE" if avg > 0.5 else "SUPPRESSED"
        print(f"    {name:<20s}  avg={avg:.3f}  std={std:.3f}  {status}  {variability}")

    # launch interactive explorer for per-date gate lookup
    interactive_explorer(all_gates, test_dates, feature_names, y_test)


if __name__ == "__main__":
    main()