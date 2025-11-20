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
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x)
        x = x + attn_out
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
        
        # Project point features to hidden dimension
        self.point_feat_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Tracking head for iterative point tracking (used for frames [0, T-1])
        self.tracking_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Predict (x, y) offset
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output head for point prediction (used for frame T)
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
    
    def get_sinusoidal_positional_encoding(self, positions, d_model):
        """
        Generate sinusoidal positional encodings for given positions
        Args:
            positions: (B, seq_len) tensor of position indices (can be any integer values)
            d_model: dimension of the model
        Returns:
            pos_encoding: (B, seq_len, d_model) positional encodings
        """
        B, seq_len = positions.shape
        device = positions.device
        
        # Convert positions to float for computation
        pos = positions.float()  # (B, seq_len)
        
        # Create dimension indices [0, 1, 2, ..., d_model//2 - 1]
        dim_indices = torch.arange(d_model // 2, dtype=torch.float32, device=device)  # (d_model//2,)
        
        # Compute the divisor: 10000^(2i/d_model)
        div_term = torch.exp(dim_indices * -(torch.log(torch.tensor(10000.0)) / d_model))  # (d_model//2,)
        
        # Compute positional encodings
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        # pos: (B, seq_len) -> (B, seq_len, 1)
        # div_term: (d_model//2,) -> (1, 1, d_model//2)
        pos_expanded = pos.unsqueeze(-1)  # (B, seq_len, 1)
        div_term_expanded = div_term.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model//2)
        angle = pos_expanded * div_term_expanded  # (B, seq_len, d_model//2)
        
        pos_encoding = torch.zeros(B, seq_len, d_model, device=device)  # (B, seq_len, d_model)
        pos_encoding[:, :, 0::2] = torch.sin(angle)  # even indices: sin
        pos_encoding[:, :, 1::2] = torch.cos(angle)  # odd indices: cos
        
        return pos_encoding
    
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
        
        # Extract frame features for all frames [0, T]
        frame_features = self.feature_extractor(frames)  # (B, T, C_feat, H_feat, W_feat)
        
        # Track points through frames [0, T-1] to get point features
        # Vectorized tracking: process all frames in a batched manner
        # Initialize with initial points
        tracked_points = initial_points.unsqueeze(1)  # (B, 1, N, 2)
        
        # Track points frame by frame - still sequential but more efficient
        for t in range(T - 1):
            current_frame_points = tracked_points[:, -1, :, :]  # (B, N, 2) - last frame's points
            current_frame_features = frame_features[:, t, :, :, :]  # (B, C_feat, H_feat, W_feat)
            
            # Extract point features at current frame
            current_points_expanded = current_frame_points.unsqueeze(1)  # (B, 1, N, 2)
            current_frame_features_expanded = current_frame_features.unsqueeze(1)  # (B, 1, C_feat, H_feat, W_feat)
            
            point_feat = self.extract_point_features(
                current_frame_features_expanded, 
                current_points_expanded
            )  # (B, 1, N, C_feat)
            
            # Reshape for processing
            point_feat_flat = point_feat.reshape(B, N, self.feature_dim)  # (B, N, C_feat)
            
            # Project point features to hidden dimension
            point_feat_proj = self.point_feat_proj(point_feat_flat)  # (B, N, hidden_dim)
            
            # Predict offset for next frame
            offset = self.tracking_head(point_feat_proj)  # (B, N, 2)
            next_points = current_frame_points + offset
            tracked_points = torch.cat([tracked_points, next_points.unsqueeze(1)], dim=1)  # (B, t+2, N, 2)
        
        # Now we have tracked points for frames [0, T-1] (T frames total)
        # Vectorized extraction of point features for frames [0, T-2]
        # tracked_points: (B, T, N, 2)
        # frame_features: (B, T, C_feat, H_feat, W_feat)
        
        # Extract point features for all frames [0, T-2] in one batch
        frame_feat_history = frame_features[:, :T-1, :, :, :]  # (B, T-1, C_feat, H_feat, W_feat)
        points_history = tracked_points[:, :T-1, :, :]  # (B, T-1, N, 2)
        
        # Vectorized point feature extraction
        point_features_history = self.extract_point_features(
            frame_feat_history, 
            points_history
        )  # (B, T-1, N, C_feat)
        
        # Get frame features for frame T (the last frame, index T-1 in 0-indexed)
        # We'll use frame features from the last frame for prediction
        frame_feat_T = frame_features[:, T-1, :, :, :]  # (B, C_feat, H_feat, W_feat)
        
        # Get the last tracked points (from frame T-1) as starting point for frame T
        last_points = tracked_points[:, -1, :, :]  # (B, N, 2) - points at frame T-1
        
        # Extract point features at frame T using last tracked points
        # (We sample at last known positions to get context)
        frame_feat_T_expanded = frame_feat_T.unsqueeze(1)  # (B, 1, C_feat, H_feat, W_feat)
        last_points_expanded = last_points.unsqueeze(1)  # (B, 1, N, 2)
        point_feat_T = self.extract_point_features(
            frame_feat_T_expanded, 
            last_points_expanded
        )  # (B, 1, N, C_feat)
        
        # Combine point features from [0, T-1] with point features at T
        # Flatten temporal and point dimensions for point features history
        point_features_history_flat = point_features_history.reshape(B, (T-1) * N, self.feature_dim)  # (B, (T-1)*N, C_feat)
        point_feat_T_flat = point_feat_T.reshape(B, N, self.feature_dim)  # (B, N, C_feat)
        
        # Project point features to hidden dimension
        point_feat_history_proj = self.point_feat_proj(point_features_history_flat)  # (B, (T-1)*N, hidden_dim)
        point_feat_T_proj = self.point_feat_proj(point_feat_T_flat)  # (B, N, hidden_dim)
        
        # Add point embeddings for history
        point_coords_history = tracked_points[:, :T-1, :, :].reshape(B, (T-1) * N, 2)  # (B, (T-1)*N, 2)
        point_embeds_history = self.point_embed(point_coords_history)  # (B, (T-1)*N, hidden_dim)
        
        # Add point embeddings for frame T
        point_embeds_T = self.point_embed(last_points)  # (B, N, hidden_dim)
        
        # Add temporal positional encodings
        temporal_positions_history = torch.arange(T-1, device=frames.device).unsqueeze(1).repeat(1, N).reshape(1, -1).repeat(B, 1)  # (B, (T-1)*N)
        temporal_embeds_history = self.get_sinusoidal_positional_encoding(
            temporal_positions_history, self.hidden_dim
        )  # (B, (T-1)*N, hidden_dim)
        
        temporal_positions_T = torch.full((B, N), T-1, device=frames.device)  # (B, N)
        temporal_embeds_T = self.get_sinusoidal_positional_encoding(
            temporal_positions_T, self.hidden_dim
        )  # (B, N, hidden_dim)
        
        # Combine all features
        x_history = point_feat_history_proj + point_embeds_history + temporal_embeds_history  # (B, (T-1)*N, hidden_dim)
        x_T = point_feat_T_proj + point_embeds_T + temporal_embeds_T  # (B, N, hidden_dim)
        
        # Concatenate history and current frame features
        x = torch.cat([x_history, x_T], dim=1)  # (B, T*N, hidden_dim)
        
        # Process through transformer
        attention_weights_list = []
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)
            attention_weights_list.append(attn_weights)
        
        # Predict point offsets for frame T (only for the last N tokens)
        x_T_out = x[:, -N:, :]  # (B, N, hidden_dim) - last N tokens correspond to frame T
        offsets_T = self.output_head(x_T_out)  # (B, N, 2)
        
        # Add offsets to last tracked points to get refined predictions for frame T-1 (last frame)
        predicted_points_T = last_points + offsets_T  # (B, N, 2)
        
        # Combine tracked points [0, T-2] with refined prediction for frame T-1
        # tracked_points: (B, T, N, 2) contains all tracked points
        all_tracked_points = tracked_points[:, :T-1, :, :]  # (B, T-1, N, 2) - exclude last frame
        predicted_points_T_expanded = predicted_points_T.unsqueeze(1)  # (B, 1, N, 2)
        predicted_points = torch.cat([all_tracked_points, predicted_points_T_expanded], dim=1)  # (B, T, N, 2)
        
        if return_attention:
            return predicted_points, attention_weights_list
        return predicted_points
