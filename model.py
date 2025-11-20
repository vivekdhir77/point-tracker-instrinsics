"""
Point Tracker Model with Attention Mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
    """
    CNN feature extractor using pre-trained ResNet backbone
    Can use ResNet18, ResNet34, or ResNet50
    """
    
    def __init__(self, in_channels=3, out_dim=256, backbone='resnet18', pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        
        # Load pre-trained ResNet backbone
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            backbone_dim = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from 'resnet18', 'resnet34', 'resnet50'")
        
        # Remove the final fully connected layer and average pooling
        # We want to keep spatial dimensions
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Project to desired output dimension
        if backbone_dim != out_dim:
            self.projection = nn.Sequential(
                nn.Conv2d(backbone_dim, out_dim, 1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.projection = nn.Identity()
        
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        # Extract features using backbone
        features = self.backbone(x)  # (B*T, backbone_dim, H', W')
        
        # Project to desired dimension
        features = self.projection(features)  # (B*T, out_dim, H', W')
        
        # Reshape back
        _, C_out, H_out, W_out = features.shape
        features = features.view(B, T, C_out, H_out, W_out)
        
        return features


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
        dropout=0.1,
        backbone='resnet18',
        pretrained=True
    ):
        super().__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor with pre-trained backbone
        self.feature_extractor = FeatureExtractor(
            in_channels=3, 
            out_dim=feature_dim,
            backbone=backbone,
            pretrained=pretrained
        )
        
        # Point feature embedding
        self.point_embed = nn.Linear(2, hidden_dim)  # (x, y) -> hidden_dim
        
        # Temporal embedding
        self.temporal_embed = nn.Embedding(1000, hidden_dim)
        
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
        
        # Reshape for grid_sample
        features_flat = features.reshape(B * T, C, H, W)
        points_flat = points.reshape(B * T, N, 1, 2)
        
        # Normalize to [-1, 1] for grid_sample (from [0, 1] normalized coordinates)
        points_normalized = points_flat * 2 - 1
        
        # Sample features using bilinear interpolation
        sampled_features = F.grid_sample(
            features_flat,
            points_normalized,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # (B*T, C, N, 1)
        
        sampled_features = sampled_features.squeeze(-1).transpose(1, 2)  # (B*T, N, C)
        sampled_features = sampled_features.reshape(B, T, N, C)
        
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
        point_features = point_features.reshape(B, T * N, self.feature_dim)
        
        # Add point embeddings
        point_coords = current_points.reshape(B, T * N, 2)
        point_embeds = self.point_embed(point_coords)
        
        # Add temporal embeddings
        temporal_ids = torch.arange(T, device=frames.device).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        temporal_ids = temporal_ids.repeat(B, 1, N)  # (B, T, N)
        temporal_ids = temporal_ids.reshape(B, T * N)  # (B, T * N)
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
        offsets = offsets.reshape(B, T, N, 2)
        
        # Add offsets to current points
        predicted_points = current_points + offsets
        
        if return_attention:
            return predicted_points, attention_weights_list
        return predicted_points

