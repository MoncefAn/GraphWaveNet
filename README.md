# Graph WaveNet — Traffic Forecasting (METR-LA)

**Mini-project (GNNs & Network Science)**  
**Baseline:** Graph WaveNet (paper reproduction) · **Optimized:** architectural & engineering improvements · **Streamlit demo** using optimized predictions



## Overview

This repository implements Graph WaveNet for multi-step traffic forecasting on the METR-LA dataset. It is a compact university mini-project demonstrating spatio-temporal graph neural nets and network science concepts.

Included:
- Baseline Graph WaveNet implementation (paper reproduction).
- **Optimized GraphWaveNet** variant (LayerNorm after GCN, stronger skip pooling, static + adaptive adjacency blending, masked losses, other fixes).
- Robust training script `Train.py` (pretrain + fine-tune workflow, LayerNorm-ready, logging, checkpointing, `--test-only`).
- Dataset generator `dataset_generator.py` that fits scaler on *training* split only (no leakage), saves sliding-window `.npz` files, adjacency `.npy`, and `scaler.pkl`.
- `compare_with_stats_and_plots.py` for multi-seed comparisons and horizon analysis.
- Streamlit dashboard `traffic_dashboard.py` to visualize optimized model predictions on a map and time series.
- Utilities: dataloaders (`data_loader.py`, `fixed_data_loader.py`), plotting and metrics helpers.



### Training Script (`Train.py`)
- Works with LayerNorm version of GraphWaveNet.
- Proper logging (JSON + CSV via `MetricsLogger`) and training plots saved to `./plots/`.
- Checkpointing: saves best model as `*_best.pt` (includes config & optimizer state).
- `--test-only` mode: loads most recent `*_best.pt`, runs evaluation, writes test & horizon metrics (15/30/60 min).
- Pretrain + fine-tune pipeline:
  - Stage 1 — pretrain on short horizon (pred_len=6 → 30min).
  - Stage 2 — transfer weights (except final projection) and finetune on full horizon (pred_len=12 → 60min).
- Masked MAE / MAPE losses (ignore NaNs and optional `null_val`).
- Deterministic seeds set (Python / NumPy / PyTorch).

### Dataset generator (`dataset_generator.py`)
- Fits `StandardScaler` on the **training period only** (no leakage).
- Produces chronological sliding-window `train.npz`, `val.npz`, `test.npz` (format: `x`, `y`, offsets).
- Saves `adj_matrix.npy` and `scaler.pkl` for reproducible inverse transforms.
- Adds temporal features: time-of-day (optional day-of-week).

### Streamlit Dashboard (`traffic_dashboard.py`)
- Interactive map (Plotly Mapbox), comparison view (pred / truth / error), per-sensor time series, error analysis.
- Loads `final_predictions_*.npz`, `scaler.pkl` and `graph_sensor_locations.csv`.
- Metrics computed using same robust masking logic as training.



## Quick start (recommended workflow)

1. Create virtualenv & install dependencies:
```bash
python -m venv .venv
```
# activate .venv
pip install -r requirements.txt

2. Generate processed data (edit paths in dataset_generator.py to point to your local METR-LA raw files):

python dataset_generator.py
# -> creates processed files under the configured output folder (e.g. processed_new/)


3. Full training (pretrain + fine-tune):

python Train.py


4. Test only (evaluate saved best model):

python Train.py --test-only


5. Multi-seed comparison & horizon plots:

python compare.py



6. Streamlit dashboard (ensure file paths inside app point to your saved predictions / scaler):

streamlit run traffic_dashboard.py


Outputs:

Checkpoints: ./saved_models/

Plots: ./plots/

Logs: ./logs/

## Common troubleshooting

If training stalls, set num_workers=0 and/or reduce batch_size.

If memory spikes: reduce batch_size, lower num_workers, or reduce model sizes.

If scaler.inverse_transform fails: ensure scaler.pkl is saved and that you reshape arrays to 2-D before inverse_transform. Robust fallbacks are implemented in the code.

If your test MAE is higher than the paper (~3.60 for 60-min on METR-LA), likely causes:

Preprocessing differences (global vs per-node scaler), or leakage.

Different hyperparameters (filters, clipping, LR schedule).

Implementation ordering differences (self-loop addition before / after normalization).

Random seeds.


## references
Graph WaveNet (original)
Z. Wu, et al., Graph WaveNet for Deep Spatial-Temporal Graph Modeling, arXiv:1906.00121.
PDF: https://arxiv.org/pdf/1906.00121

Improvements & optimizations
S. Shleifer, C. McCreery, V. Chitters, Incrementally Improving Graph WaveNet Performance on Traffic Prediction, arXiv:1912.07390 (Dec 2019).
PDF: https://arxiv.org/pdf/1912.07390


::contentReference[oaicite:0]{index=0}


