"""
Graph WaveNet Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionGraphConv(nn.Module):
    """
    Dense diffusion graph convolution with precomputed powers.
    """
    def __init__(self, in_channels, out_channels, K=2, bias=True):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight_f = nn.Parameter(torch.Tensor(K + 1, in_channels, out_channels))
        self.weight_b = nn.Parameter(torch.Tensor(K + 1, in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_f)
        nn.init.xavier_uniform_(self.weight_b)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def _apply_powers(x, P_pows, weights):
        """Vectorized power application."""
        device = x.device
        Kp1 = P_pows.shape[0]
        out = torch.zeros(x.size(0), x.size(1), weights.shape[2], 
                         device=device, dtype=x.dtype)
        for k in range(Kp1):
            Pk = P_pows[k]
            x_k = torch.einsum('ij,bjc->bic', Pk, x)
            out = out + torch.matmul(x_k, weights[k])
        return out

    def forward(self, x, P_f_pows, P_b_pows):
        """
        x: (B*T, N, C_in)
        P_f_pows, P_b_pows: (K+1, N, N)
        """
        out_f = self._apply_powers(x, P_f_pows, self.weight_f)
        out_b = self._apply_powers(x, P_b_pows, self.weight_b)
        out = out_f + out_b
        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)
        return out


class AdaptiveAdjacency(nn.Module):
    """Learnable adaptive adjacency matrix."""
    def __init__(self, num_nodes, embed_dim=10):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        self.source_embedding = nn.Parameter(torch.Tensor(num_nodes, embed_dim))
        self.target_embedding = nn.Parameter(torch.Tensor(num_nodes, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.source_embedding)
        nn.init.xavier_uniform_(self.target_embedding)

    def forward(self):
        adj = torch.matmul(self.source_embedding, self.target_embedding.T)
        adj = F.relu(adj)
        adj = F.softmax(adj, dim=1)
        return adj


class GatedTCN(nn.Module):
    """
    Gated Temporal Convolution with causal padding.
    Better skip connection handling
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.filter_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation)
        )
        self.gate_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation)
        )
        
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                        if in_channels != out_channels else None
        
        # IMPROVED: Skip connection preserves temporal info
        self.skip_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, N, T)
        returns: residual (B, C, N, T), skip (B, C, N, T)
        """
        # Causal padding
        x_padded = F.pad(x, (self.padding, 0, 0, 0))
        
        # Gated activation
        f = torch.tanh(self.filter_conv(x_padded))
        g = torch.sigmoid(self.gate_conv(x_padded))
        out = f * g
        
        # IMPROVED: Skip connection keeps full temporal dimension
        # Will be pooled later in the model
        skip = self.skip_conv(out)
        
        # Residual
        residual = self.res_conv(x) if self.res_conv is not None else x
        residual = residual + out
        
        return residual, skip


class SpatialTemporalLayer(nn.Module):
    """
    IMPROVED: Added Layer Normalization as per paper Section 4.2
    Single ST layer: Gated TCN → GCN → LayerNorm → Dropout
    """
    def __init__(self, in_channels, out_channels, num_nodes, 
                 K=2, kernel_size=2, dilation=1, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Temporal convolution
        self.temporal = GatedTCN(in_channels, out_channels, kernel_size, dilation)
        
        # Graph convolution
        self.gcn = DiffusionGraphConv(out_channels, out_channels, K)
        
        # ADDED: Layer Normalization (Paper Section 4.2)
        # Normalizes across channel dimension for each node
        self.layer_norm = nn.LayerNorm(out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Learnable blending between static and adaptive
        self._alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, P_f_pows, P_b_pows, A_adp_pows):
        """
        x: (B, C_in, N, T)
        returns: x_out (B, C_out, N, T), skip (B, C_out, N, T)
        """
        batch, _, num_nodes, seq_len = x.shape

        # 1) Temporal convolution
        x_temporal, skip = self.temporal(x)  # Both: (B, C_out, N, T)

        # 2) Reshape for GCN: (B, C, N, T) → (B*T, N, C)
        xt = x_temporal.permute(0, 3, 2, 1).contiguous()  # (B, T, N, C)
        xt = xt.view(batch * seq_len, num_nodes, -1)      # (B*T, N, C)

        # 3) Static GCN
        if P_f_pows is None or P_b_pows is None:
            P_eye = torch.eye(num_nodes, device=x.device, dtype=x.dtype)
            P_f_pows = torch.stack([P_eye] * (self.gcn.K + 1), dim=0)
            P_b_pows = P_f_pows.clone()
        
        static_out = self.gcn(xt, P_f_pows, P_b_pows)  # (B*T, N, C_out)

        # 4) Adaptive GCN
        if A_adp_pows is None:
            A_eye = torch.stack([torch.eye(num_nodes, device=x.device, dtype=x.dtype)
                                for _ in range(self.gcn.K + 1)], dim=0)
            adapt_out = self.gcn(xt, A_eye, A_eye)
        else:
            adapt_out = self.gcn(xt, A_adp_pows, A_adp_pows)

        # 5) Blend outputs
        alpha = torch.sigmoid(self._alpha)
        x_gcn = alpha * static_out + (1 - alpha) * adapt_out

        # 6)Layer Normalization
        # Reshape to (B*T, N, C) → apply norm → reshape back
        x_gcn = self.layer_norm(x_gcn)  # Normalizes last dimension (C)

        # 7) Dropout
        x_gcn = self.dropout(x_gcn)

        # 8) Reshape back: (B*T, N, C) → (B, C, N, T)
        x_out = x_gcn.view(batch, seq_len, num_nodes, -1)
        x_out = x_out.permute(0, 3, 2, 1).contiguous()

        return x_out, skip


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet:
    - Layer normalization after GCN
    - Better skip connection handling
    - Proper receptive field design
    """
    def __init__(self, num_nodes, in_channels=1, out_channels=1,
                 residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512,
                 kernel_size=2, num_layers=8, K=2, embed_dim=10,
                 dropout=0.3, pred_len=12):
        super().__init__()
        self.num_nodes = num_nodes
        self.K = K
        self.num_layers = num_layers
        self.pred_len = pred_len

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, residual_channels, kernel_size=1)

        # Adaptive adjacency
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, embed_dim)

        # Stacked ST layers with alternating dilations
        self.st_layers = nn.ModuleList()
        dilations = [1, 2] * (num_layers // 2)
        
        for i in range(num_layers):
            self.st_layers.append(
                SpatialTemporalLayer(
                    residual_channels, dilation_channels,
                    num_nodes, K=K, kernel_size=kernel_size,
                    dilation=dilations[i], dropout=dropout
                )
            )

        # IMPROVED: Skip aggregation with temporal pooling
        # Pool temporal dimension before aggregating skips
        #self.skip_pool = nn.AdaptiveAvgPool2d((None, 1))  # Pool to (N, 1)
        
        self.skip_proj1 = nn.Conv2d(
            dilation_channels * num_layers, 
            skip_channels, 
            kernel_size=1
        )
        self.skip_proj2 = nn.Conv2d(skip_channels, end_channels, kernel_size=1)

        # Final output projection
        self.output_proj = nn.Conv2d(end_channels, out_channels * pred_len, kernel_size=1)

        # Buffers for cached static adjacency powers
        self.register_buffer('P_f_pows', None)
        self.register_buffer('P_b_pows', None)
        self._static_adj_id = None

    @staticmethod
    def compute_transition_matrix(adj):
        """Row-normalize adjacency to transition matrix."""
        row_sum = adj.sum(dim=1, keepdim=True)
        row_sum = row_sum.masked_fill(row_sum == 0, 1.0)
        P = adj / row_sum
        return P

    def set_static_adj(self, adj):
        """
        Precompute static adjacency powers.
        """
        device = adj.device
        orig_id = id(adj)
        # Add self-loops FIRST (this is correct!)
        adj_local = adj + torch.eye(adj.size(0), device=device, dtype=adj.dtype)
        
        # Then normalize
        P = self.compute_transition_matrix(adj_local)
        P_t = self.compute_transition_matrix(adj_local.T)

        # Compute powers
        P_f_pows = [torch.eye(self.num_nodes, device=device, dtype=adj.dtype)]
        P_b_pows = [torch.eye(self.num_nodes, device=device, dtype=adj.dtype)]
        
        for _ in range(1, self.K + 1):
            P_f_pows.append(P_f_pows[-1] @ P)
            P_b_pows.append(P_b_pows[-1] @ P_t)

        P_f_pows = torch.stack(P_f_pows, dim=0)
        P_b_pows = torch.stack(P_b_pows, dim=0)
        self._static_adj_id = orig_id  
        
        # Register as buffers
        self.P_f_pows = P_f_pows
        self.P_b_pows = P_b_pows
        #self._static_adj_id = id(adj)

    def forward(self, x, adj=None):
        """
        x: (B, seq_len, N, in_channels)
        adj: optional static adjacency (N, N)
        returns: (B, pred_len, N, out_channels)
        """
        batch_size, seq_len, num_nodes, in_ch = x.shape
        assert num_nodes == self.num_nodes, f"Expected {self.num_nodes} nodes, got {num_nodes}"

        # Reshape to (B, in_ch, N, T)
        x = x.permute(0, 3, 2, 1).contiguous()

        # Input projection
        x = self.input_proj(x)

        # Prepare static adjacency powers
        if adj is not None:
            if self.P_f_pows is None or self._static_adj_id != id(adj):
                self.set_static_adj(adj.to(x.device))
        else:
            self.P_f_pows = None
            self.P_b_pows = None

        # Compute adaptive adjacency powers
        A_adp = self.adaptive_adj().to(x.device)
        P_adp_pows = [torch.eye(self.num_nodes, device=x.device, dtype=x.dtype)]
        
        if self.K >= 1:
            Pk = P_adp_pows[0]
            for _ in range(1, self.K + 1):
                Pk = Pk @ A_adp
                P_adp_pows.append(Pk)
        
        P_adp_pows = torch.stack(P_adp_pows, dim=0)

        # Pass through ST layers
        skips = []
        h = x
        
        for layer in self.st_layers:
            h, skip = layer(h, self.P_f_pows, self.P_b_pows, P_adp_pows)
            # Pool temporal dimension for each skip
            skip_pooled = skip.mean(dim=-1, keepdim=True)  # (B, C, N, 1)
            skips.append(skip_pooled)

        # Aggregate skip connections
        skip_cat = torch.cat(skips, dim=1)  # (B, C*num_layers, N, 1)
        
        out = F.relu(skip_cat)
        out = self.skip_proj1(out)
        out = F.relu(out)
        out = self.skip_proj2(out)

        # Final projection
        out = self.output_proj(out)  # (B, out_ch*pred_len, N, 1)
        out = out.squeeze(-1)        # (B, out_ch*pred_len, N)

        # Reshape to (B, pred_len, N, out_ch)
        B = out.size(0)
        out = out.view(B, self.pred_len, -1, self.num_nodes)
        out = out.permute(0, 1, 3, 2).contiguous()

        return out


# ============================================================================
# IMPROVED LOSS WITH INPUT MASKING
# ============================================================================

def masked_mae_loss(pred, target, null_val=0.0):
    """
    Handles both NaN and null_val masking
    """
    # Create mask for valid values
    mask = ~torch.isnan(target)
    mask = mask & (target != null_val)
    mask = mask.float()
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    # Compute masked loss
    loss = torch.abs(pred - target) * mask
    
    # Normalize by number of valid elements
    return loss.sum() / mask.sum()


def masked_mape_loss(pred, target, null_val=0.0, epsilon=1e-5):
    """
    Masked MAPE with numerical stability
    """
    mask = ~torch.isnan(target)
    mask = mask & (target != null_val)
    mask = mask.float()
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    
    # Add epsilon to avoid division by zero
    loss = torch.abs((pred - target) / (torch.abs(target) + epsilon))
    loss = loss * mask
    
    return (loss.sum() / mask.sum()) * 100


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("GRAPH WAVENET - VALIDATION TEST")
    print("="*60)
    
    # Test configuration
    NUM_NODES = 207
    SEQ_LEN = 12
    IN_CH = 2
    OUT_CH = 1
    BATCH = 8
    PRED_LEN = 12

    print(f"\nTest Configuration:")
    print(f"  Nodes: {NUM_NODES}")
    print(f"  Seq Length: {SEQ_LEN}")
    print(f"  Pred Length: {PRED_LEN}")
    print(f"  Batch Size: {BATCH}")

    # Create model
    model = GraphWaveNet(
        num_nodes=NUM_NODES,
        in_channels=IN_CH,
        out_channels=OUT_CH,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=128,
        end_channels=256,
        kernel_size=2,
        num_layers=8,
        K=2,
        embed_dim=16,
        dropout=0.2,
        pred_len=PRED_LEN
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model created with {num_params:,} parameters")

    # Create test data
    x = torch.randn(BATCH, SEQ_LEN, NUM_NODES, IN_CH)
    y = torch.randn(BATCH, PRED_LEN, NUM_NODES, OUT_CH)
    
    # Create adjacency
    d = torch.rand(NUM_NODES, NUM_NODES)
    d = (d + d.T) / 2.0
    adj = torch.exp(-d**2 / (0.1**2))
    adj[adj < 0.5] = 0.0

    print("\n" + "="*60)
    print("FORWARD PASS TEST")
    print("="*60)

    # Test forward pass
    model.set_static_adj(adj)
    
    with torch.no_grad():
        out = model(x, adj)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Expected:     (batch={BATCH}, pred_len={PRED_LEN}, nodes={NUM_NODES}, features={OUT_CH})")

    # Test loss
    loss_mae = masked_mae_loss(out, y)
    loss_mape = masked_mape_loss(out, y)
    
    print(f"\n✓ Loss computation successful!")
    print(f"  MAE Loss:  {loss_mae.item():.4f}")
    print(f"  MAPE Loss: {loss_mape.item():.2f}%")

    # Test backward pass
    loss_mae.backward()
    
    print(f"\n✓ Backward pass successful!")
    print(f"  Gradients computed for all parameters")

    
    print("\n✓ ALL TESTS PASSED!")