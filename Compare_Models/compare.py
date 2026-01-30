"""
Compare two trained GraphWaveNet checkpoints across MULTIPLE horizons.

- Overall metrics (MAE, RMSE, MAPE)
- Horizon-wise MAE
- Multi-seed evaluation (mean ± std)
- Saves CSV + plots
NO TRAINING — evaluation only
"""

import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ======================================================
# USER CONFIG
# ======================================================
DATA_DIR = "../dcrnn_data-main/metr_la/processed_new"
MODEL_DIR = "./models"
OUT_DIR = "./comparison_outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN = 12
BATCH_SIZE = 16
FULL_PRED_LEN = 12

# Horizons (in timesteps)
HORIZONS = {
    "15min": 3,
    "30min": 6,
    "60min": 12
}

SEEDS = [42, 123, 999]

BASELINE_CKPT = "graphwavenet_20260124_161031_best.pt"
OPTIMIZED_CKPT = "graphwavenet_20260125_175953_best.pt"

# ======================================================
# MODEL CONFIGS (MUST MATCH TRAINING)
# ======================================================
COMMON_CFG = dict(
    NUM_NODES=207,
    IN_CHANNELS=2,
    OUT_CHANNELS=1,
    K=2,
    EMBED_DIM=10,
    kernel_size=2
)

BASELINE_CFG = dict(
    RESIDUAL_CHANNELS=32,
    DILATION_CHANNELS=32,
    SKIP_CHANNELS=256,
    END_CHANNELS=512,
    NUM_LAYERS=8,
    DROPOUT=0.3
)

OPTIMIZED_CFG = dict(
    RESIDUAL_CHANNELS=40,
    DILATION_CHANNELS=40,
    SKIP_CHANNELS=256,
    END_CHANNELS=512,
    NUM_LAYERS=8,
    DROPOUT=0.3
)

# ======================================================
# IMPORTS FROM YOUR CODEBASE
# ======================================================
from fixed_data_loader import create_dataloaders
from WaveNet import GraphWaveNet
from Train import evaluate


# ======================================================
# HELPERS
# ======================================================
def masked_metrics(y_true, y_pred, tiny_threshold=0.1, rel_eps=1e-2):
    """
    Computes MAE / RMSE / MAPE while ignoring NaNs.

    MAPE is computed robustly:
      - exclude entries where |y_true| < tiny_threshold (too small to produce meaningful relative error)
      - use denom = max(|y_true|, eps) where eps = max(rel_eps * mean(|y_true|), 1e-3)
    Returns: (mae, rmse, mape) where mape is a percentage (or nan if not computable).
    """
    # ignore NaNs in either array
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if y_true.size == 0:
        return float('nan'), float('nan'), float('nan')

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Robust MAPE: exclude tiny true values
    mask_no_tiny = np.abs(y_true) >= tiny_threshold
    if mask_no_tiny.sum() == 0:
        mape = float('nan')
    else:
        t_nt = y_true[mask_no_tiny]
        p_nt = y_pred[mask_no_tiny]
        mean_abs = np.mean(np.abs(t_nt))
        eps = max(rel_eps * mean_abs, 1e-3)
        denom = np.maximum(np.abs(t_nt), eps)
        mape = float(np.mean(np.abs((t_nt - p_nt) / denom)) * 100.0)

    return mae, rmse, mape



def safe_torch_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def build_model(cfg, pred_len):
    return GraphWaveNet(
        num_nodes=COMMON_CFG["NUM_NODES"],
        in_channels=COMMON_CFG["IN_CHANNELS"],
        out_channels=COMMON_CFG["OUT_CHANNELS"],
        residual_channels=cfg["RESIDUAL_CHANNELS"],
        dilation_channels=cfg["DILATION_CHANNELS"],
        skip_channels=cfg["SKIP_CHANNELS"],
        end_channels=cfg["END_CHANNELS"],
        kernel_size=COMMON_CFG["kernel_size"],
        num_layers=cfg["NUM_LAYERS"],
        K=COMMON_CFG["K"],
        embed_dim=COMMON_CFG["EMBED_DIM"],
        dropout=cfg["DROPOUT"],
        pred_len=pred_len
    ).to(DEVICE)


def inverse_transform(arr, scaler):
    if scaler is None:
        return arr
    flat = arr.reshape(-1, 1)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(arr.shape)


def horizon_mae(preds, targets, scaler):
    y_p = inverse_transform(preds[..., 0], scaler)
    y_t = inverse_transform(targets[..., 0], scaler)

    H = y_t.shape[1]
    maes = []

    for h in range(H):
        yt = y_t[:, h, :].reshape(-1)
        yp = y_p[:, h, :].reshape(-1)

        mask = ~np.isnan(yt)
        maes.append(np.mean(np.abs(yt[mask] - yp[mask])))

    return np.array(maes)



def overall_metrics(preds, targets, scaler):
    y_p = inverse_transform(preds[..., 0], scaler)
    y_t = inverse_transform(targets[..., 0], scaler)

    y_p = y_p.flatten()
    y_t = y_t.flatten()

    return masked_metrics(y_t, y_p)



# ======================================================
# EVALUATION
# ======================================================
def evaluate_model(
    cfg,
    ckpt,
    horizon_len,
    test_loader,
    adj,
    scaler,
    seed
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    #  Always build model with FULL_PRED_LEN
    model = build_model(cfg, FULL_PRED_LEN)
    model.set_static_adj(adj.to(DEVICE))

    state = safe_torch_load(os.path.join(MODEL_DIR, ckpt))
    model.load_state_dict(
        state["model_state_dict"] if "model_state_dict" in state else state
    )
    model.eval()

    _, _, _, preds, targets = evaluate(
        model, test_loader, adj.to(DEVICE), DEVICE, scaler, mode="Test"
    )

    #  Slice horizon
    preds = preds[:, :horizon_len]
    targets = targets[:, :horizon_len]

    mae, rmse, mape = overall_metrics(preds, targets, scaler)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "horizon_mae": horizon_mae(preds, targets, scaler)
    }


# ======================================================
# MAIN
# ======================================================
def main():
    all_results = []

    # Data loader created ONCE (full pred_len)
    _, _, test_loader, adj, scaler = create_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        pred_len=FULL_PRED_LEN
    )

    for horizon_name, horizon_len in HORIZONS.items():
        print(f"\n=== Evaluating horizon: {horizon_name} ({horizon_len} steps) ===")

        for model_name, cfg, ckpt in [
            ("Baseline", BASELINE_CFG, BASELINE_CKPT),
            ("Optimized", OPTIMIZED_CFG, OPTIMIZED_CKPT)
        ]:
            seed_results = []

            for seed in SEEDS:
                res = evaluate_model(
                    cfg, ckpt, horizon_len,
                    test_loader, adj, scaler, seed
                )
                seed_results.append(res)

            maes = np.array([r["MAE"] for r in seed_results])
            rmses = np.array([r["RMSE"] for r in seed_results])
            mapes = np.array([r["MAPE"] for r in seed_results])
            horizon_stack = np.stack([r["horizon_mae"] for r in seed_results])

            all_results.append({
                "model": model_name,
                "horizon": horizon_name,
                "MAE_mean": maes.mean(),
                "MAE_std": maes.std(),
                "RMSE_mean": rmses.mean(),
                "RMSE_std": rmses.std(),
                "MAPE_mean": mapes.mean(),
                "MAPE_std": mapes.std()
            })

            # Plot horizon-wise MAE
            x = np.arange(1, horizon_len + 1)
            plt.figure(figsize=(8, 4))
            plt.plot(x, horizon_stack.mean(0), marker="o")
            plt.fill_between(
                x,
                horizon_stack.mean(0) - horizon_stack.std(0),
                horizon_stack.mean(0) + horizon_stack.std(0),
                alpha=0.25
            )
            plt.title(f"{model_name} – {horizon_name}")
            plt.xlabel("Horizon step")
            plt.ylabel("MAE")
            plt.grid(alpha=0.3)
            plt.tight_layout()

            plot_path = os.path.join(
                OUT_DIR, f"horizon_{model_name}_{horizon_name}.png"
            )
            plt.savefig(plot_path, dpi=200)
            plt.close()

    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(
        OUT_DIR,
        f"metrics_multi_horizon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df.to_csv(csv_path, index=False)
    print("\n✅ Saved results to:", csv_path)


if __name__ == "__main__":
    main()
