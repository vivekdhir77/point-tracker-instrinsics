"""
Point Tracker Model with Attention Mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out, attn


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward"""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x)
        x = x + attn_out
        
        # Feed-forward
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class FeatureExtractor(nn.Module):
    """CNN feature extractor for image frames"""
    
    def __init__(self, in_channels=3, out_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, out_dim, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_dim)
        
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Reshape back
        _, C_out, H_out, W_out = x.shape
        x = x.view(B, T, C_out, H_out, W_out)
        
        return x


class PointTracker(nn.Module):
    """
    Point Tracker Model using Transformer architecture
    Tracks points across video frames using attention mechanism
    """
    
    def __init__(
        self,
        feature_dim=256,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        num_points=8,
        dropout=0.1
    ):
        super().__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(in_channels=3, out_dim=feature_dim)
        
        # Point feature embedding
        self.point_embed = nn.Linear(2, hidden_dim)  # (x, y) -> hidden_dim
        
        # Temporal embedding
        self.temporal_embed = nn.Embedding(1000, hidden_dim)  # Support up to 1000 frames
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output head for point prediction
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Predict (x, y) offset
        )
        
    def extract_point_features(self, features, points):
        """
        Extract features at given point locations using bilinear interpolation
        Args:
            features: (B, T, C, H, W) feature maps
            points: (B, T, N, 2) normalized point coordinates [0, 1]
        Returns:
            point_features: (B, T, N, C) features at point locations
        """
        B, T, C, H, W = features.shape
        N = points.shape[2]
        
        # Convert normalized coordinates to pixel coordinates
        points_pixel = points.clone()
        points_pixel[:, :, :, 0] = points_pixel[:, :, :, 0] * (W - 1)
        points_pixel[:, :, :, 1] = points_pixel[:, :, :, 1] * (H - 1)
        
        # Reshape for grid_sample
        features_flat = features.view(B * T, C, H, W)
        points_flat = points_pixel.view(B * T, N, 1, 2)
        
        # Normalize to [-1, 1] for grid_sample
        points_normalized = points_flat.clone()
        points_normalized[:, :, :, 0] = (points_normalized[:, :, :, 0] / (W - 1)) * 2 - 1
        points_normalized[:, :, :, 1] = (points_normalized[:, :, :, 1] / (H - 1)) * 2 - 1
        
        # Sample features using bilinear interpolation
        sampled_features = F.grid_sample(
            features_flat,
            points_normalized,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # (B*T, C, N, 1)
        
        sampled_features = sampled_features.squeeze(-1).transpose(1, 2)  # (B*T, N, C)
        sampled_features = sampled_features.view(B, T, N, C)
        
        return sampled_features
    
    def forward(self, frames, initial_points, return_attention=False):
        """
        Forward pass
        Args:
            frames: (B, T, C, H, W) video frames
            initial_points: (B, N, 2) initial point locations (normalized [0, 1])
            return_attention: whether to return attention weights
        Returns:
            predicted_points: (B, T, N, 2) predicted point locations
            attention_weights: list of attention weights if return_attention=True
        """
        B, T, C, H, W = frames.shape
        N = initial_points.shape[1]
        
        # Extract features
        features = self.feature_extractor(frames)  # (B, T, C_feat, H_feat, W_feat)
        
        # Initialize point trajectories
        current_points = initial_points.unsqueeze(1).repeat(1, T, 1, 1)  # (B, T, N, 2)
        
        # Extract point features
        point_features = self.extract_point_features(features, current_points)  # (B, T, N, C)
        
        # Flatten temporal and point dimensions
        point_features = point_features.view(B, T * N, self.feature_dim)
        
        # Add point embeddings
        point_coords = current_points.view(B, T * N, 2)
        point_embeds = self.point_embed(point_coords)
        
        # Add temporal embeddings
        temporal_ids = torch.arange(T, device=frames.device).unsqueeze(0).unsqueeze(-1)
        temporal_ids = temporal_ids.repeat(1, 1, N).view(B, T * N)
        temporal_embeds = self.temporal_embed(temporal_ids)
        
        # Combine embeddings
        x = point_features + point_embeds + temporal_embeds
        
        # Process through transformer
        attention_weights_list = []
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)
            attention_weights_list.append(attn_weights)
        
        # Predict point offsets
        offsets = self.output_head(x)  # (B, T*N, 2)
        offsets = offsets.view(B, T, N, 2)
        
        # Add offsets to current points
        predicted_points = current_points + offsets
        
        # Clamp to valid range
        predicted_points = torch.clamp(predicted_points, 0, 1)
        
        if return_attention:
            return predicted_points, attention_weights_list
        return predicted_points

