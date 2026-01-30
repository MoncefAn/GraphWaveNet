# PYG_fixed_data_loader.py  (replace existing load/create functions with these)

import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    """(unchanged) expects a time-series array shaped (T, N, F) and slices windows."""
    def __init__(self, data, seq_len=12, pred_len=12):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = len(data) - seq_len - pred_len + 1
    def __len__(self):
        return max(0, self.num_samples)
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class SequenceDataset(Dataset):
    """New: accepts precomputed sequences: x shape (S, seq_len, N, F), y shape (S, pred_len, N, F)."""
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0], "x and y must have same number of samples"
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.y[idx])

def load_metr_la_data(data_dir):
    """
    Loads dataset supporting two formats:
     - 'raw' time-series format: train.npz contains key 'data' -> shape (T, N, F)
     - 'sliding-window' format: train.npz contains 'x' and 'y' -> shapes (S, seq_len, N, F) and (S, pred_len, N, F)

    Returns:
        If sliding-window: returns (x_train, y_train), (x_val, y_val), (x_test, y_test), adj_matrix, scaler, mode='seq'
        If raw: returns train_data, val_data, test_data, adj_matrix, scaler, mode='ts'
    """
    train_np = np.load(os.path.join(data_dir, 'train.npz'))
    val_np = np.load(os.path.join(data_dir, 'val.npz'))
    test_np = np.load(os.path.join(data_dir, 'test.npz'))

    # Try to detect format
    if 'data' in train_np.files:
        # raw time-series mode
        train_data = train_np['data']  # (T, N, F)
        val_data = val_np['data']
        test_data = test_np['data']
        mode = 'ts'
    elif 'x' in train_np.files and 'y' in train_np.files:
        # sliding-window precomputed mode
        x_train, y_train = train_np['x'], train_np['y']
        x_val, y_val = val_np['x'], val_np['y']
        x_test, y_test = test_np['x'], test_np['y']
        mode = 'seq'
        train_data, val_data, test_data = (x_train, y_train), (x_val, y_val), (x_test, y_test)
    else:
        raise RuntimeError(f"train.npz keys: {train_np.files}. Expected 'data' or ('x','y').")

    # Load adjacency
    adj_path = os.path.join(data_dir, 'adj_matrix.npy')
    if os.path.exists(adj_path):
        adj_matrix = np.load(adj_path).astype(np.float32)
    else:
        # fallback to adj_mx.pkl if present
        pkl_path = os.path.join(os.path.dirname(data_dir), 'adj_mx.pkl')
        if os.path.exists(pkl_path):
            import pickle as pkl
            with open(pkl_path, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
            adj_matrix = np.array(data[2], dtype=np.float32)
        else:
            raise FileNotFoundError("No adjacency found (adj_matrix.npy or adj_mx.pkl)")

    # Load scaler if exists
    scaler_path = os.path.join(data_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = None

    return train_data, val_data, test_data, adj_matrix, scaler, mode


def create_dataloaders(data_dir, batch_size=16, seq_len=12, pred_len=12, num_workers=0):
    """
    Universal dataloader creator that supports both raw time-series and precomputed sequences.
    Returns train_loader, val_loader, test_loader, adj_matrix (torch.FloatTensor), scaler
    """
    train_data, val_data, test_data, adj_matrix, scaler, mode = load_metr_la_data(data_dir)

    if mode == 'ts':
        # train_data are full timeseries (T,N,F). create TrafficDataset (slicer)
        train_dataset = TrafficDataset(train_data, seq_len=seq_len, pred_len=pred_len)
        val_dataset = TrafficDataset(val_data, seq_len=seq_len, pred_len=pred_len)
        test_dataset = TrafficDataset(test_data, seq_len=seq_len, pred_len=pred_len)
    else:
        # mode == 'seq': train_data is (x_train, y_train)
        x_train, y_train = train_data
        x_val, y_val = val_data
        x_test, y_test = test_data

        # Ensure the precomputed windows match requested pred_len
        def align_y(y_array, desired_len):
            cur_len = y_array.shape[1]
            if cur_len == desired_len:
                return y_array
            elif cur_len > desired_len:
                # slice to desired horizon
                return y_array[:, :desired_len, ...]
            else:
                # pad with NaNs so losses/metrics can mask them
                pad_shape = (y_array.shape[0], desired_len - cur_len, y_array.shape[2], y_array.shape[3])
                pad = np.full(pad_shape, np.nan, dtype=y_array.dtype)
                return np.concatenate([y_array, pad], axis=1)

        y_train = align_y(y_train, pred_len)
        y_val   = align_y(y_val,   pred_len)
        y_test  = align_y(y_test,  pred_len)

        # Optionally align x as well if someone precomputed x with differing seq_len (defensive)
        def align_x(x_array, desired_seq_len):
            cur = x_array.shape[1]
            if cur == desired_seq_len:
                return x_array
            elif cur > desired_seq_len:
                return x_array[:, -desired_seq_len:, ...]  # keep last seq_len timesteps
            else:
                # pad at front with NaNs (rare)
                pad_shape = (x_array.shape[0], desired_seq_len - cur, x_array.shape[2], x_array.shape[3])
                pad = np.full(pad_shape, np.nan, dtype=x_array.dtype)
                return np.concatenate([pad, x_array], axis=1)

        x_train = align_x(x_train, seq_len)
        x_val   = align_x(x_val,   seq_len)
        x_test  = align_x(x_test,  seq_len)

        train_dataset = SequenceDataset(x_train, y_train)
        val_dataset   = SequenceDataset(x_val,   y_val)
        test_dataset  = SequenceDataset(x_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=False)

    adj_matrix = torch.FloatTensor(adj_matrix)

    print(f"Data mode: {mode}")
    if mode == 'seq':
        print(f"Train samples: {len(train_dataset)} (x shape {x_train.shape}, y shape {y_train.shape})")
    else:
        print(f"Train time-series length: {train_data.shape}")

    return train_loader, val_loader, test_loader, adj_matrix, scaler
