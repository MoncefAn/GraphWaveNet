"""
Fixed Training Script for Graph WaveNet
- Works with LayerNorm version
- Proper logging and checkpointing
- Test-only mode support
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Import your modules
from fixed_data_loader import create_dataloaders
from WaveNet import GraphWaveNet


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    DATA_DIR = "../dcrnn_data-main/metr_la/processed"
    
    # Model
    NUM_NODES = 207  # METR-LA
    IN_CHANNELS = 2  # [speed, time_of_day]
    OUT_CHANNELS = 1  # predict speed only
    RESIDUAL_CHANNELS = 40 # Increased from 32
    DILATION_CHANNELS = 40 # Increased from 32
    SKIP_CHANNELS = 256
    END_CHANNELS = 512
    NUM_LAYERS = 8
    K = 2  # Diffusion steps
    EMBED_DIM = 10
    
    # Training
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    SEQ_LEN = 12
    PRED_LEN = 12
    DROPOUT = 0.3
    CLIP_GRAD = 3.0 # Reduced from 5.0
    
    # Early stopping
    PATIENCE = 15
    
    # Logging & saving
    LOG_DIR = "./logs"
    MODEL_DIR = "./saved_models"
    PLOT_DIR = "./plots"
    EXPERIMENT_NAME = f"graphwavenet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directories
    for d in [LOG_DIR, MODEL_DIR, PLOT_DIR]:
        os.makedirs(d, exist_ok=True)


cfg = Config()


# ============================================================================
# METRICS LOGGER
# ============================================================================

class MetricsLogger:
    """Logs training metrics to JSON and CSV"""
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_mape': [],
            'learning_rate': [],
            'timestamp': []
        }
        self.json_file = os.path.join(log_dir, f"{experiment_name}.json")
        self.csv_file = os.path.join(log_dir, f"{experiment_name}.csv")
        
    def log(self, epoch, train_loss, val_mae, val_rmse, val_mape, lr):
        """Log metrics for one epoch"""
        timestamp = datetime.now().isoformat()
        
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(float(train_loss))
        self.metrics['val_mae'].append(float(val_mae))
        self.metrics['val_rmse'].append(float(val_rmse))
        self.metrics['val_mape'].append(float(val_mape))
        self.metrics['learning_rate'].append(float(lr))
        self.metrics['timestamp'].append(timestamp)
        
        # Save to JSON
        with open(self.json_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save to CSV
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.csv_file, index=False)
        
    def plot_metrics(self, save_dir):
        """Generate and save training curves"""
        df = pd.DataFrame(self.metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Metrics: {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # Train loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', 
                       color='#2E86AB', linewidth=2)
        axes[0, 0].set_title('Training Loss (MAE)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Validation MAE
        axes[0, 1].plot(df['epoch'], df['val_mae'], label='Val MAE', 
                       color='#A23B72', linewidth=2)
        axes[0, 1].set_title('Validation MAE', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Validation RMSE
        axes[1, 0].plot(df['epoch'], df['val_rmse'], label='Val RMSE', 
                       color='#F18F01', linewidth=2)
        axes[1, 0].set_title('Validation RMSE', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Validation MAPE
        axes[1, 1].plot(df['epoch'], df['val_mape'], label='Val MAPE', 
                       color='#C73E1D', linewidth=2)
        axes[1, 1].set_title('Validation MAPE', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_dir, f"{self.experiment_name}_metrics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved: {plot_path}")
        plt.close()


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def masked_mae_loss(pred, target, null_val=None):
    """MAE loss with masking for NaNs and null values"""
    mask = ~torch.isnan(target)
    if null_val is not None:
        mask = mask & (target != null_val)
    mask = mask.float()
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    loss = torch.abs(pred - target) * mask
    return loss.sum() / mask.sum()


def compute_metrics(y_true, y_pred, scaler=None, tiny_threshold=0.1, rel_eps=1e-2):
    """
    Compute MAE, RMSE, MAPE with robust handling
    
    Returns:
        mae, rmse, mape (floats)
    """
    # Extract speed channel (feature 0)
    y_true_s = y_true[..., 0].astype(np.float64)
    y_pred_s = y_pred[..., 0].astype(np.float64)

    # Inverse transform if scaler provided
    if scaler is not None:
        S, H, N = y_true_s.shape
        flat_t = y_true_s.reshape(-1, 1)
        flat_p = y_pred_s.reshape(-1, 1)
        try:
            inv_t = scaler.inverse_transform(flat_t).reshape(S, H, N)
            inv_p = scaler.inverse_transform(flat_p).reshape(S, H, N)
            y_true_inv = inv_t
            y_pred_inv = inv_p
        except Exception as e:
            print(f"Warning: scaler.inverse_transform failed: {e}")
            y_true_inv = y_true_s.copy()
            y_pred_inv = y_pred_s.copy()
    else:
        y_true_inv = y_true_s.copy()
        y_pred_inv = y_pred_s.copy()

    # Flatten and mask NaNs
    flat_t = y_true_inv.flatten()
    flat_p = y_pred_inv.flatten()
    valid = ~np.isnan(flat_t) & ~np.isnan(flat_p)
    flat_t = flat_t[valid]
    flat_p = flat_p[valid]
    
    if flat_t.size == 0:
        return float('nan'), float('nan'), float('nan')

    # MAE
    mae = float(mean_absolute_error(flat_t, flat_p))
    
    # RMSE
    try:
        rmse = float(mean_squared_error(flat_t, flat_p, squared=False))
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(flat_t, flat_p)))

    # MAPE (excluding tiny values)
    mask_no_tiny = np.abs(flat_t) >= tiny_threshold
    if mask_no_tiny.sum() == 0:
        mape = float('nan')
    else:
        t_nt = flat_t[mask_no_tiny]
        p_nt = flat_p[mask_no_tiny]
        mean_abs = np.mean(np.abs(t_nt))
        eps = max(rel_eps * mean_abs, 1e-3)
        denom = np.maximum(np.abs(t_nt), eps)
        mape = float(np.mean(np.abs((t_nt - p_nt) / denom)) * 100.0)

    return mae, rmse, mape


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, loader, optimizer, criterion, adj_matrix, device, scheduler):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)  # (B, seq_len, N, F)
        batch_y = batch_y.to(device)  # (B, pred_len, N, F)
        
        # Replace NaNs with zeros for gradient computation
        #batch_y_clean = torch.where(torch.isnan(batch_y), 
        #                           torch.zeros_like(batch_y), 
        #                            batch_y)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(batch_x, adj_matrix)  # (B, pred_len, N, out_ch)
        
        # Compute loss (only on speed channel if multi-feature)
        if batch_y.shape[-1] > 1:
            #loss = criterion(pred[..., 0], batch_y_clean[..., 0])
            loss = criterion(pred[...,0], batch_y[...,0])
        else:
            loss = criterion(pred, batch_y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item() * batch_x.size(0)
        num_samples += batch_x.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_samples


def evaluate(model, loader, adj_matrix, device, scaler=None, mode='Val'):
    """Evaluate model and return metrics + predictions"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc=f'{mode} Eval', leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_x, adj_matrix)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    mae, rmse, mape = compute_metrics(all_targets, all_preds, scaler=scaler)
    
    return mae, rmse, mape, all_preds, all_targets


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_predictions(preds, targets, save_path, num_samples=3, num_nodes=3):
    """Plot sample predictions vs targets"""
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Plot multiple nodes
        time_steps = range(preds.shape[1])  # pred_len
        
        for node_idx in range(min(num_nodes, preds.shape[2])):
            pred_node = preds[i, :, node_idx, 0]
            target_node = targets[i, :, node_idx, 0]
            
            # Skip if all NaN
            if not np.isnan(target_node).all():
                ax.plot(time_steps, pred_node, 
                       label=f'Pred Node {node_idx}', 
                       marker='o', linewidth=2, alpha=0.7)
                ax.plot(time_steps, target_node, 
                       label=f'True Node {node_idx}', 
                       marker='s', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_title(f'Sample {i+1}: Predictions vs Ground Truth', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step (5-min intervals)')
        ax.set_ylabel('Speed')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Prediction plot saved: {save_path}")

# ============================================================================
# pretraining on shorter horizons and fine-tuning on the full horizon    
# ============================================================================
# ============================================================================
# MAIN TRAINING
# ============================================================================
def pretrain_and_finetune(model, train_loader, val_loader, test_loader, adj_matrix, device, cfg, scaler):
    """
    Two-stage training: pretrain on shorter horizon, then finetune on full horizon
    FIXED: Properly unpack 5 return values from create_dataloaders
    """
    print("\n" + "="*80)
    print("STAGE 1: PRETRAINING ON SHORTER HORIZON (30 minutes)")
    print("="*80)
    
    # Create dataloaders for shorter horizon (6 timesteps = 30 minutes)
    pretrain_train_loader, pretrain_val_loader, pretrain_test_loader, pretrain_adj_matrix, pretrain_scaler = create_dataloaders(
        cfg.DATA_DIR,
        batch_size=cfg.BATCH_SIZE,
        seq_len=cfg.SEQ_LEN,
        pred_len=6  # 30 minutes instead of 60
    )
    
    pretrain_adj_matrix = pretrain_adj_matrix.to(device)
    
    # Create new model for pretraining (with pred_len=6)
    pretrain_model = GraphWaveNet(
        num_nodes=cfg.NUM_NODES,
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        residual_channels=cfg.RESIDUAL_CHANNELS,
        dilation_channels=cfg.DILATION_CHANNELS,
        skip_channels=cfg.SKIP_CHANNELS,
        end_channels=cfg.END_CHANNELS,
        kernel_size=2,
        num_layers=cfg.NUM_LAYERS,
        K=cfg.K,
        embed_dim=cfg.EMBED_DIM,
        dropout=cfg.DROPOUT,
        pred_len=6  # Shorter horizon
    ).to(device)
    
    pretrain_model.set_static_adj(pretrain_adj_matrix)
    
    # Pretrain optimizer and scheduler
    pretrain_optimizer = torch.optim.Adam(
        pretrain_model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    pretrain_scheduler = torch.optim.lr_scheduler.StepLR(
        pretrain_optimizer, 
        step_size=1, 
        gamma=0.97
    )
    
    # Pretrain for 60 epochs
    best_pretrain_mae = float('inf')
    pretrain_epochs = 60
    
    for epoch in range(pretrain_epochs):
        # Train
        train_loss = train_one_epoch(
            pretrain_model, 
            pretrain_train_loader, 
            pretrain_optimizer, 
            masked_mae_loss, 
            pretrain_adj_matrix, 
            device, 
            pretrain_scheduler
        )
        
        # Validate
        val_mae, val_rmse, val_mape, _, _ = evaluate(
            pretrain_model, 
            pretrain_val_loader, 
            pretrain_adj_matrix, 
            device, 
            pretrain_scaler, 
            'Val'
        )
        
        # Track best
        if val_mae < best_pretrain_mae:
            best_pretrain_mae = val_mae
            status = "✓ BEST"
        else:
            status = ""
        
        print(f"Pretrain Epoch {epoch+1:3d}/{pretrain_epochs} - "
              f"Loss: {train_loss:.4f} | "
              f"Val MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.2f}% "
              f"{status}")
    
    print(f"\n✓ Pretraining complete. Best Val MAE: {best_pretrain_mae:.4f}")
    
    # ========== Transfer learned weights to full model ==========
    print("\n" + "="*80)
    print("STAGE 2: FINE-TUNING ON FULL HORIZON (60 minutes)")
    print("="*80)
    
    # Copy weights from pretrained model (excluding final projection layer)
    pretrained_state = pretrain_model.state_dict()
    model_state = model.state_dict()
    
    # Copy all weights except output_proj (which has different dimensions)
    transferred_keys = []
    for key in pretrained_state:
        if 'output_proj' not in key and key in model_state:
            if pretrained_state[key].shape == model_state[key].shape:
                model_state[key] = pretrained_state[key]
                transferred_keys.append(key)
    
    model.load_state_dict(model_state)
    print(f"✓ Transferred {len(transferred_keys)} parameter tensors from pretrained model")
    
    # Fine-tune optimizer and scheduler
    fine_tune_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE * 0.5,  # Lower learning rate for fine-tuning
        weight_decay=cfg.WEIGHT_DECAY
    )
    fine_tune_scheduler = torch.optim.lr_scheduler.StepLR(
        fine_tune_optimizer, 
        step_size=1, 
        gamma=0.97
    )
    
    # Fine-tune for 40 epochs
    best_finetune_mae = float('inf')
    finetune_epochs = 40
    best_model_path = os.path.join(cfg.MODEL_DIR, f"{cfg.EXPERIMENT_NAME}_best.pt")
    
    for epoch in range(finetune_epochs):
        # Train
        train_loss = train_one_epoch(
            model, 
            train_loader, 
            fine_tune_optimizer, 
            masked_mae_loss, 
            adj_matrix, 
            device, 
            fine_tune_scheduler
        )
        
        # Validate
        val_mae, val_rmse, val_mape, _, _ = evaluate(
            model, 
            val_loader, 
            adj_matrix, 
            device, 
            scaler, 
            'Val'
        )
        
        # Save best model
        if val_mae < best_finetune_mae:
            best_finetune_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': fine_tune_optimizer.state_dict(),
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_mape': val_mape
            }, best_model_path)
            status = "✓ BEST - saved"
        else:
            status = ""
        
        print(f"Finetune Epoch {epoch+1:3d}/{finetune_epochs} - "
              f"Loss: {train_loss:.4f} | "
              f"Val MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.2f}% "
              f"{status}")
    
    print(f"\n✓ Fine-tuning complete. Best Val MAE: {best_finetune_mae:.4f}")
    print(f"✓ Best model saved: {best_model_path}")
    
    # ========== Final Test Evaluation ==========
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_mae, test_rmse, test_mape, test_preds, test_targets = evaluate(
        model, test_loader, adj_matrix, device, scaler, 'Test'
    )
    
    print(f"\n{'='*80}")
    print("FINAL TEST RESULTS")
    print(f"{'='*80}")
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAPE: {test_mape:.2f}%")
    print(f"{'='*80}")
    
    # Save predictions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pred_file = os.path.join(cfg.PLOT_DIR, f"final_predictions_{timestamp}.npz")
    np.savez_compressed(pred_file, predictions=test_preds, targets=test_targets)
    print(f"\n✓ Predictions saved: {pred_file}")
    
    # Plot sample predictions
    plot_path = os.path.join(cfg.PLOT_DIR, f"final_samples_{timestamp}.png")
    plot_predictions(test_preds, test_targets, plot_path)


def main():
    print("="*80)
    print("GRAPH WAVENET TRAINING - PRETRAIN + FINETUNE")
    print("="*80)
    print(f"Device: {cfg.DEVICE}")
    print(f"Experiment: {cfg.EXPERIMENT_NAME}")
    
    # ========== Load Data ==========
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_loader, val_loader, test_loader, adj_matrix, scaler = create_dataloaders(
        cfg.DATA_DIR,
        batch_size=cfg.BATCH_SIZE,
        seq_len=cfg.SEQ_LEN,
        pred_len=cfg.PRED_LEN,
        num_workers=0
    )
    
    adj_matrix = adj_matrix.to(cfg.DEVICE)
    
    # Verify shapes
    for x, y in train_loader:
        print(f"✓ Data shapes verified:")
        print(f"  Input:  {x.shape}")
        print(f"  Target: {y.shape}")
        print(f"  Adjacency: {adj_matrix.shape}")
        break
    
    # ========== Create Model ==========
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = GraphWaveNet(
        num_nodes=cfg.NUM_NODES,
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        residual_channels=cfg.RESIDUAL_CHANNELS,
        dilation_channels=cfg.DILATION_CHANNELS,
        skip_channels=cfg.SKIP_CHANNELS,
        end_channels=cfg.END_CHANNELS,
        kernel_size=2,
        num_layers=cfg.NUM_LAYERS,
        K=cfg.K,
        embed_dim=cfg.EMBED_DIM,
        dropout=cfg.DROPOUT,
        pred_len=cfg.PRED_LEN
    ).to(cfg.DEVICE)
    
    model.set_static_adj(adj_matrix)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")
    
    # ========== Pretrain and Fine-tune ==========
    pretrain_and_finetune(model, train_loader, val_loader, test_loader, 
                         adj_matrix, cfg.DEVICE, cfg, scaler)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)


# ============================================================================
# TEST ONLY MODE
# ============================================================================

def test_only():
    """Run test evaluation without training"""
    print("="*80)
    print("TEST-ONLY MODE")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, adj_matrix, scaler = create_dataloaders(
        cfg.DATA_DIR, batch_size=cfg.BATCH_SIZE, 
        seq_len=cfg.SEQ_LEN, pred_len=cfg.PRED_LEN
    )
    adj_matrix = adj_matrix.to(cfg.DEVICE)
    
    # Create model
    print("Creating model...")
    model = GraphWaveNet(
        num_nodes=cfg.NUM_NODES,
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        residual_channels=cfg.RESIDUAL_CHANNELS,
        dilation_channels=cfg.DILATION_CHANNELS,
        skip_channels=cfg.SKIP_CHANNELS,
        end_channels=cfg.END_CHANNELS,
        kernel_size=2,
        num_layers=cfg.NUM_LAYERS,
        K=cfg.K,
        embed_dim=cfg.EMBED_DIM,
        dropout=cfg.DROPOUT,
        pred_len=cfg.PRED_LEN
    ).to(cfg.DEVICE)
    
    model.set_static_adj(adj_matrix)
    
    # Find best model
    print("\nLooking for best model...")
    best_files = [f for f in os.listdir(cfg.MODEL_DIR) if f.endswith('_best.pt')]
    
    if not best_files:
        print("❌ No best model found. Please train first.")
        return
    
    # Use most recent
    best_files.sort(reverse=True)
    best_path = os.path.join(cfg.MODEL_DIR, best_files[0])
    
    checkpoint = torch.load(best_path, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded: {best_files[0]}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val MAE: {checkpoint['val_mae']:.4f}")
    
    # Test
    print("\nRunning test evaluation...")
    test_mae, test_rmse, test_mape, test_preds, test_targets = evaluate(
        model, test_loader, adj_matrix, cfg.DEVICE, scaler, 'Test'
    )
    
    print(f"\n{'='*80}")
    print("TEST RESULTS")
    print(f"{'='*80}")
    print(f"MAE:  {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAPE: {test_mape:.2f}%")
    print(f"{'='*80}")
    
    # Save predictions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pred_file = os.path.join(cfg.PLOT_DIR, f"test_predictions_{timestamp}.npz")
    np.savez_compressed(pred_file, predictions=test_preds, targets=test_targets)
    print(f"\n✓ Predictions saved: {pred_file}")
    
    # Plot
    plot_path = os.path.join(cfg.PLOT_DIR, f"test_samples_{timestamp}.png")
    plot_predictions(test_preds, test_targets, plot_path)
    
    print("\n TEST COMPLETE")    


if __name__ == "__main__":
    if '--test-only' in sys.argv or '--test' in sys.argv:
        test_only()
    else:
        main()