# dataset_generator.py - corrected / annotated version
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def generate_graph_seq2seq_io_data(
    df: pd.DataFrame, 
    x_offsets: np.ndarray, 
    y_offsets: np.ndarray,
    add_time_in_day: bool = True,
    add_day_in_week: bool = False,
    scaler: StandardScaler = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Generate sequences with temporal features (official Graph WaveNet format)
    Returns:
        x: (num_samples, input_length, num_nodes, input_dim)
        y: (num_samples, output_length, num_nodes, output_dim)
        scaler: scaler passed in or fitted (if None, fitted on provided df)
    Note: This function **does not** split train/val/test — splitting must be chronological outside.
    """
    num_samples, num_nodes = df.shape
    
    # Main traffic data (speed) as primary feature
    data = np.expand_dims(df.values, axis=-1)  # (T, N, 1)
    features = [data]
    
    # Add time-of-day feature in a clearer way
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        # time_ind shape: (T,), we want shape (T, N, 1)
        time_in_day = np.repeat(time_ind.reshape(-1, 1), repeats=num_nodes, axis=1)[..., np.newaxis]
        features.append(time_in_day)
    
    # Add day-of-week feature (optional)
    if add_day_in_week:
        dow = df.index.dayofweek.values  # shape (T,)
        dow_tiled = np.repeat(dow.reshape(-1, 1), repeats=num_nodes, axis=1)[..., np.newaxis]
        features.append(dow_tiled)
    
    # Concatenate features -> (T, N, F)
    data = np.concatenate(features, axis=-1)
    
    # Scaling: if scaler is provided, use it; otherwise fit one.
    # IMPORTANT: scaler should be fit on training data to avoid leakage.
    if scaler is None:
        scaler = StandardScaler()
        reshaped = data.reshape(-1, data.shape[-1])
        scaler.fit(reshaped[:, 0].reshape(-1, 1))
    
    # Apply scaler to the primary feature across whole series using the provided/fitted scaler
    reshaped = data.reshape(-1, data.shape[-1])
    reshaped[:, 0] = scaler.transform(reshaped[:, 0].reshape(-1, 1)).flatten()
    data = reshaped.reshape(data.shape)
    
    # Build sliding windows: use explicit start/end indices to avoid sign confusion:
    start = int(-min(x_offsets))                      # earliest t to sample
    end = int(num_samples - max(y_offsets))          # first t that would spill beyond data
    
    x, y = [], []
    for t in range(start, end):
        x.append(data[t + np.array(x_offsets), ...])
        y.append(data[t + np.array(y_offsets), ...])
    
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y, scaler


def load_adjacency_matrix(pkl_path: str, output_dir: str):
    """
    Load DCRNN adjacency matrix and save as .npy for easier loading
    
    Args:
        pkl_path: Path to adj_mx.pkl
        output_dir: Directory to save adj_matrix.npy
    """
    print(f"Loading adjacency matrix from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    adj_matrix = np.array(data[2], dtype=np.float32)
    
    output_path = os.path.join(output_dir, 'adj_matrix.npy')
    np.save(output_path, adj_matrix)
    print(f"✓ Saved adjacency matrix to {output_path}")
    print(f"  Shape: {adj_matrix.shape}")
    print(f"  Sparsity: {(adj_matrix == 0).sum() / adj_matrix.size:.2%}")
    return adj_matrix


def generate_dataset():
    """
    Main function to generate train/val/test dataset.
    Key change: fit scaler on training period ONLY to avoid leakage.
    """
    print("=" * 60)
    print("Graph WaveNet Dataset Generator")
    print("=" * 60)
    
    # ---- configuration  ----
    data_file = "../dcrnn_data-main/metr_la/vel_metr_la.h5"
    adj_file = "../dcrnn_data-main/adj_mx.pkl"
    output_dir = "../dcrnn_data-main/metr_la/processed_new"
    seq_len = 12
    y_start = 1
    train_ratio = 0.7
    val_ratio = 0.1
    add_day_in_week = False
    # ---------------------------------------------------------------------
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load traffic data
    print(f"\nLoading traffic data from {data_file}")
    if data_file.endswith('.h5'):
        df = pd.read_hdf(data_file)
    elif data_file.endswith('.csv'):
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        raise ValueError("Data file must be .h5 or .csv")
    
    print(f"✓ Traffic data shape: {df.shape}")
    print(f"✓ Date range: {df.index.min()} to {df.index.max()}")
    
    # Save adjacency as .npy for easier future use
    load_adjacency_matrix(adj_file, output_dir)
    
    # Offsets
    x_offsets = np.sort(np.arange(-(seq_len - 1), 1, 1))  # [-11,...,0]
    y_offsets = np.sort(np.arange(y_start, seq_len + 1, 1))  # [1,...,12]
    
    print(f"\nGenerating sequences...")
    print(f"  Input length: {seq_len}, Prediction length: {seq_len}")
    print(f"  x_offsets: {x_offsets[:3]}...{x_offsets[-3:]}")
    print(f"  y_offsets: {y_offsets[:3]}...{y_offsets[-3:]}")
    
    # ---- Fit scaler on training time range only (prevent leakage) ----
    num_total_timesteps = df.shape[0]
    num_train_timesteps = int(num_total_timesteps * train_ratio)
    train_df = df.iloc[:num_train_timesteps]
    
    # Fit scaler on train_df primary signal
    scaler = StandardScaler()
    # reshape to (T*N, 1)
    train_vals = np.expand_dims(train_df.values, axis=-1).reshape(-1, 1)
    scaler.fit(train_vals)
    # ------------------------------------------------------------------
    
    # Generate windows for entire df but using *training-fitted* scaler to transform values
    x, y, _ = generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets,
        add_time_in_day=True,
        add_day_in_week=add_day_in_week,
        scaler=scaler      # pass the pre-fitted scaler
    )
    
    print(f"✓ Generated sequences: x={x.shape}, y={y.shape}")
    print(f"  Value range (after scaling): [{x[..., 0].min():.3f}, {x[..., 0].max():.3f}]")
    
    # Chronological split on windows (not on raw timesteps). Compute number of samples:
    num_samples = x.shape[0]
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    
    train_x = x[:num_train]; train_y = y[:num_train]
    val_x = x[num_train:num_train + num_val]; val_y = y[num_train:num_train + num_val]
    test_x = x[num_train + num_val:]; test_y = y[num_train + num_val:]
    
    # Save splits
    print("\nSaving splits:")
    for split_name, (x_split, y_split) in {
        'train': (train_x, train_y),
        'val': (val_x, val_y),
        'test': (test_x, test_y)
    }.items():
        output_file = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(
            output_file,
            x=x_split,
            y=y_split,
            x_offsets=x_offsets.reshape(-1, 1),
            y_offsets=y_offsets.reshape(-1, 1)
        )
        print(f"  ✓ {split_name}: {x_split.shape}, {y_split.shape} -> {output_file}")
    
    # Save training-fitted scaler for inverse transforms
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved to {scaler_path}")
    
    print("\n" + "=" * 60)
    print(" Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    generate_dataset()
