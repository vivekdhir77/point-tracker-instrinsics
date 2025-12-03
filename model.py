"""
Point Tracker Model with Attention Mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.linalg import rq as scipy_rq


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
        pretrained=True,
        use_correlation_matching=True,  # Use correlation-based feature matching
        search_radius=4  # Search radius for correlation matching
    ):
        super().__init__()
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_correlation_matching = use_correlation_matching
        self.search_radius = search_radius
        
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
        
        # Learnable temporal token (one per frame) to aggregate frame-level information
        self.temporal_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.normal_(self.temporal_token, mean=0, std=0.02)
        
        # Correlation volume projection (if using correlation matching)
        if use_correlation_matching:
            corr_channels = (2 * search_radius + 1) ** 2
            self.corr_proj = nn.Sequential(
                nn.Linear(corr_channels, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim)  # Project to full hidden_dim
            )
        
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
        
        # Ray prediction head for all tracked points (predicts 6D Plücker coordinates)
        # Plücker coordinates: (direction d: 3D, moment m: 3D) where m = p × d
        self.ray_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 6),  # 6D Plücker coordinates: [d_x, d_y, d_z, m_x, m_y, m_z]
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
    
    def extract_local_window_features(self, features, points, window_size=7):
        """
        Extract features in a local window around given points using correlation
        This helps when the tracked point might have drifted from the true location
        
        Args:
            features: (B, T, C, H, W) feature maps
            points: (B, T, N, 2) normalized point coordinates [0, 1]
            window_size: size of the local window to extract (must be odd)
        Returns:
            window_features: (B, T, N, C, window_size, window_size) features in local window
        """
        B, T, C, H, W = features.shape
        N = points.shape[2]
        
        # Reshape for processing
        features_flat = features.reshape(B * T, C, H, W)
        points_flat = points.reshape(B * T, N, 2)
        
        # Convert normalized coordinates [0, 1] to pixel coordinates
        points_y = points_flat[:, :, 1] * H  # (B*T, N)
        points_x = points_flat[:, :, 0] * W  # (B*T, N)
        
        # Create window grid offsets
        offset = window_size // 2
        grid_y, grid_x = torch.meshgrid(
            torch.arange(-offset, offset + 1, device=features.device),
            torch.arange(-offset, offset + 1, device=features.device),
            indexing='ij'
        )  # (window_size, window_size)
        
        # Expand for all points
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)
        
        # Create sampling grid for all points
        sample_y = points_y.unsqueeze(-1).unsqueeze(-1) + grid_y  # (B*T, N, window_size, window_size)
        sample_x = points_x.unsqueeze(-1).unsqueeze(-1) + grid_x  # (B*T, N, window_size, window_size)
        
        # Normalize to [-1, 1] for grid_sample
        sample_x_norm = (sample_x / W) * 2 - 1
        sample_y_norm = (sample_y / H) * 2 - 1
        
        # Stack to create grid (B*T, N, window_size, window_size, 2)
        sampling_grid = torch.stack([sample_x_norm, sample_y_norm], dim=-1)
        
        # Sample features for each point
        window_features_list = []
        for i in range(N):
            grid_i = sampling_grid[:, i, :, :, :]  # (B*T, window_size, window_size, 2)
            sampled = F.grid_sample(
                features_flat,
                grid_i,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )  # (B*T, C, window_size, window_size)
            window_features_list.append(sampled)
        
        # Stack all points
        window_features = torch.stack(window_features_list, dim=1)  # (B*T, N, C, window_size, window_size)
        window_features = window_features.reshape(B, T, N, C, window_size, window_size)
        
        return window_features
    
    def compute_correlation_volume(self, feat_source, feat_target, points_source, search_radius=6):
        """
        Compute correlation volume between source point features and target frame features
        This allows the model to search for the point in a local neighborhood
        
        Args:
            feat_source: (B, C, H, W) source frame features
            feat_target: (B, C, H, W) target frame features
            points_source: (B, N, 2) point locations in source frame [0, 1]
            search_radius: radius of search window in pixels
        Returns:
            correlation_volume: (B, N, (2*search_radius+1)^2) correlation scores
            offset_coords: (B, N, (2*search_radius+1)^2, 2) corresponding offset coordinates
        """
        B, C, H, W = feat_source.shape
        N = points_source.shape[1]
        
        # Extract source point features
        points_expanded = points_source.unsqueeze(1)  # (B, 1, N, 2)
        feat_source_expanded = feat_source.unsqueeze(1)  # (B, 1, C, H, W)
        source_point_features = self.extract_point_features(
            feat_source_expanded,
            points_expanded
        ).squeeze(1)  # (B, N, C)
        
        # Convert points to pixel coordinates
        points_y = points_source[:, :, 1] * H  # (B, N)
        points_x = points_source[:, :, 0] * W  # (B, N)
        
        # Create search grid
        search_range = torch.arange(-search_radius, search_radius + 1, device=feat_source.device)
        grid_y, grid_x = torch.meshgrid(search_range, search_range, indexing='ij')
        grid_offsets = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (search_area, 2)
        search_area = grid_offsets.shape[0]
        
        # Compute correlation for each point
        correlations = []
        offset_coords_list = []
        
        for b in range(B):
            point_correlations = []
            point_offsets = []
            
            for n in range(N):
                # Get candidate positions
                center_x, center_y = points_x[b, n], points_y[b, n]
                candidate_x = center_x + grid_offsets[:, 0]  # (search_area,)
                candidate_y = center_y + grid_offsets[:, 1]  # (search_area,)
                
                # Normalize to [-1, 1]
                candidate_x_norm = (candidate_x / W) * 2 - 1
                candidate_y_norm = (candidate_y / H) * 2 - 1
                
                # Create sampling grid
                sampling_grid = torch.stack([candidate_x_norm, candidate_y_norm], dim=-1)  # (search_area, 2)
                sampling_grid = sampling_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, search_area, 2)
                
                # Sample features
                target_features = F.grid_sample(
                    feat_target[b:b+1],
                    sampling_grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                )  # (1, C, 1, search_area)
                target_features = target_features.squeeze(2).squeeze(0).transpose(0, 1)  # (search_area, C)
                
                # Compute correlation (cosine similarity)
                source_feat = source_point_features[b, n, :].unsqueeze(0)  # (1, C)
                source_feat_norm = F.normalize(source_feat, dim=1)
                target_feat_norm = F.normalize(target_features, dim=1)
                correlation = torch.sum(source_feat_norm * target_feat_norm, dim=1)  # (search_area,)
                
                point_correlations.append(correlation)
                point_offsets.append(grid_offsets)
            
            correlations.append(torch.stack(point_correlations, dim=0))  # (N, search_area)
            offset_coords_list.append(torch.stack(point_offsets, dim=0))  # (N, search_area, 2)
        
        correlation_volume = torch.stack(correlations, dim=0)  # (B, N, search_area)
        offset_coords = torch.stack(offset_coords_list, dim=0)  # (B, N, search_area, 2)
        
        return correlation_volume, offset_coords
    
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
    
    def normalize_ray_direction(self, ray_direction):
        """
        Normalize ray direction to unit vector
        Args:
            ray_direction: (..., 3) tensor of ray directions
        Returns:
            normalized direction: (..., 3) tensor
        """
        return F.normalize(ray_direction, dim=-1, eps=1e-6)
    
    def plucker_to_ray(self, plucker_coords):
        """
        Convert Plücker coordinates to ray representation (point, direction)
        Plücker coords: (d, m) where d is direction, m = p × d
        
        Args:
            plucker_coords: (..., 6) tensor [d_x, d_y, d_z, m_x, m_y, m_z]
        Returns:
            direction: (..., 3) normalized ray direction
            moment: (..., 3) moment vector m = p × d
        """
        direction = plucker_coords[..., :3]  # (d_x, d_y, d_z)
        moment = plucker_coords[..., 3:]     # (m_x, m_y, m_z)
        
        # Normalize direction
        direction = self.normalize_ray_direction(direction)
        
        return direction, moment
    
    def recover_camera_center_from_rays(self, ray_directions, ray_moments):
        """
        Recover camera center from ray bundle using least-squares optimization
        Equation (4): c = argmin_p Σ ||p × d - m||²
        
        This finds the 3D point closest to all rays (approximate intersection)
        
        Args:
            ray_directions: (B, M, 3) ray directions (d)
            ray_moments: (B, M, 3) ray moments (m = p × d)
        Returns:
            camera_center: (B, 3) estimated camera center in world coordinates
        """
        B, M, _ = ray_directions.shape
        device = ray_directions.device
        
        # For each batch, solve: minimize ||p × d - m||² over all rays
        # This is equivalent to solving a linear system A * p = b
        # where A and b come from linearizing the cross product constraint
        
        camera_centers = []
        for b in range(B):
            d = ray_directions[b]  # (M, 3)
            m = ray_moments[b]     # (M, 3)
            
            # Build linear system from cross product constraint
            # p × d = m, but [d]_× @ p = d × p = -p × d = -m
            # So we solve [d]_× @ p = -m, which gives us p × d = m
            # where [d]_× is the skew-symmetric matrix of d
            
            # Stack all constraints: A * p = b
            A_list = []
            b_list = []
            
            for i in range(M):
                d_i = d[i]  # (3,)
                m_i = m[i]  # (3,)
                
                # Skew-symmetric matrix [d]_× for cross product
                # [d]_× @ p = d × p = -(p × d) = -m
                d_skew = torch.tensor([
                    [0, -d_i[2], d_i[1]],
                    [d_i[2], 0, -d_i[0]],
                    [-d_i[1], d_i[0], 0]
                ], device=device, dtype=d_i.dtype)
                
                A_list.append(d_skew)
                b_list.append(-m_i)  # Note: negative sign because [d]_× @ p = -(p × d)
            
            A = torch.stack(A_list, dim=0).reshape(-1, 3)  # (3*M, 3)
            b = torch.stack(b_list, dim=0).reshape(-1)     # (3*M,)
            
            # Solve least-squares: A^T A p = A^T b
            ATA = A.T @ A  # (3, 3)
            ATb = A.T @ b  # (3,)
            
            # Add regularization for numerical stability
            ATA = ATA + 1e-6 * torch.eye(3, device=device, dtype=ATA.dtype)
            
            # Solve for camera center
            try:
                p = torch.linalg.solve(ATA, ATb)
            except:
                # Fallback to pseudoinverse if singular
                p = torch.linalg.lstsq(A, b).solution
            
            camera_centers.append(p)
        
        camera_center = torch.stack(camera_centers, dim=0)  # (B, 3)
        return camera_center
    
    def recover_rotation_intrinsics_from_rays(self, ray_directions, pixel_coords, image_size=(256, 256)):
        """
        Recover rotation R and intrinsics K from ray bundle
        Equation (5): P = argmin Σ ||H*d_i × u_i||
        
        Then decompose P via RQ decomposition: P = K @ R
        
        Args:
            ray_directions: (B, M, 3) normalized ray directions in world frame
            pixel_coords: (B, M, 2) corresponding 2D pixel coordinates [0, 1] normalized
            image_size: (H, W) image dimensions for denormalization
        Returns:
            rotation: (B, 3, 3) rotation matrix R
            intrinsics: (B, 3, 3) intrinsic matrix K
        """
        B, M, _ = ray_directions.shape
        device = ray_directions.device
        H, W = image_size
        
        rotations = []
        intrinsics_list = []
        
        for b in range(B):
            d = ray_directions[b]  # (M, 3) - world frame ray directions
            u = pixel_coords[b]    # (M, 2) - normalized pixel coords [0, 1]
            
            # Denormalize pixel coordinates to actual pixel values
            u_pixels = u.clone()
            u_pixels[:, 0] = u[:, 0] * W  # x
            u_pixels[:, 1] = u[:, 1] * H  # y
            
            # Homogeneous pixel coordinates
            u_homog = torch.cat([u_pixels, torch.ones(M, 1, device=device)], dim=1)  # (M, 3)
            
            # Solve for homography H that minimizes ||H*d_i × u_i||
            # This is a DLT (Direct Linear Transform) problem
            
            A = []
            for i in range(M):
                d_i = d[i]      # (3,)
                u_i = u_homog[i]  # (3,)
                
                # Cross product constraint: H*d_i × u_i = 0
                # This gives 3 equations, but only 2 are independent
                # We use the 2 independent equations
                
                # u × (H*d) = 0, which expands to:
                # u_y * (H_3 * d) - u_z * (H_2 * d) = 0
                # u_z * (H_1 * d) - u_x * (H_3 * d) = 0
                # u_x * (H_2 * d) - u_y * (H_1 * d) = 0
                
                # Build matrix rows for DLT
                zeros = torch.zeros(3, device=device, dtype=d_i.dtype)
                
                # Row 1: u_y * (H_3 * d) - 1 * (H_2 * d) = 0 (assuming u_z = 1 for homogeneous)
                A.append(torch.cat([
                    zeros,                    # -H_1 coefficient
                    -u_i[2] * d_i,           # -H_2 coefficient  
                    u_i[1] * d_i             # H_3 coefficient
                ]))
                
                # Row 2: 1 * (H_1 * d) - u_x * (H_3 * d) = 0
                A.append(torch.cat([
                    u_i[2] * d_i,            # H_1 coefficient
                    zeros,                    # -H_2 coefficient
                    -u_i[0] * d_i            # -H_3 coefficient
                ]))
            
            A = torch.stack(A, dim=0)  # (2*M, 9)
            
            # Solve using SVD: A * h = 0 (homogeneous system)
            # Solution is the right singular vector corresponding to smallest singular value
            try:
                _, _, Vh = torch.linalg.svd(A)
                h = Vh[-1, :]  # Last row of V^H (smallest singular value)
            except:
                # Fallback
                h = torch.randn(9, device=device)
            
            # Reshape to 3x3 homography matrix
            H = h.reshape(3, 3)
            
            # Normalize H
            H = H / (torch.norm(H) + 1e-8)
            
            # Decompose H = K @ R using RQ decomposition
            # RQ decomposition: H = K @ R where K is upper triangular, R is orthonormal
            K, R = self.rq_decomposition(H)
            
            # Ensure K has positive diagonal (conventional)
            signs = torch.sign(torch.diag(K))
            signs[signs == 0] = 1
            K = K @ torch.diag(signs)
            R = torch.diag(signs) @ R
            
            # Ensure R is a proper rotation (det(R) = 1, not -1)
            if torch.det(R) < 0:
                R = -R
                K = -K
            
            # Normalize K so that K[2,2] = 1
            K = K / K[2, 2]
            
            rotations.append(R)
            intrinsics_list.append(K)
        
        rotation = torch.stack(rotations, dim=0)  # (B, 3, 3)
        intrinsics = torch.stack(intrinsics_list, dim=0)  # (B, 3, 3)
        
        return rotation, intrinsics
    
    def rq_decomposition(self, M):
        """
        RQ decomposition: M = R @ Q where R is upper triangular, Q is orthonormal
        Uses scipy.linalg.rq for robust decomposition
        
        Args:
            M: (3, 3) torch tensor or numpy array
        Returns:
            R: (3, 3) upper triangular matrix (torch tensor)
            Q: (3, 3) orthonormal matrix (torch tensor)
        """
        # Convert to numpy if torch tensor
        if isinstance(M, torch.Tensor):
            M_np = M.detach().cpu().numpy()
            device = M.device
            dtype = M.dtype
        else:
            M_np = M
            device = None
            dtype = None
        
        # Use scipy's RQ decomposition
        R_np, Q_np = scipy_rq(M_np)
        
        # Convert back to torch if needed
        if device is not None:
            R = torch.from_numpy(R_np).to(device=device, dtype=dtype)
            Q = torch.from_numpy(Q_np).to(device=device, dtype=dtype)
        else:
            R = R_np
            Q = Q_np
        
        return R, Q
    
    def compute_translation_from_rotation_center(self, rotation, camera_center):
        """
        Compute translation vector from rotation and camera center
        Equation: t = -R * c
        
        Camera projection: P_cam = R @ P_world + t
        At the camera center C (in world coords): R @ C + t = 0
        Therefore: t = -R @ C
        
        Args:
            rotation: (B, 3, 3) rotation matrix
            camera_center: (B, 3) camera center in world coordinates
        Returns:
            translation: (B, 3) translation vector
        """
        # t = -R @ C
        translation = -torch.bmm(rotation, camera_center.unsqueeze(-1)).squeeze(-1)
        return translation
    
    def forward(self, frames, initial_points, return_attention=False, return_rays=False, return_camera=False):
        """
        Forward pass
        
        Args:
            frames: (B, T, C, H, W) video frames
            initial_points: (B, N, 2) initial point locations (normalized [0, 1])
            return_attention: whether to return attention weights
            return_rays: whether to return predicted rays (Plücker coordinates)
            return_camera: whether to recover and return camera parameters (center, rotation, intrinsics, translation)
        
        Returns:
            dict with keys:
                'points': (B, T, N, 2) predicted point locations (always present)
                'rays': (B, T, N, 6) predicted rays in Plücker coordinates (if return_rays=True)
                'camera': dict with camera parameters (if return_camera=True)
                    - 'center': (B, T, 3) camera center in world coordinates
                    - 'rotation': (B, T, 3, 3) rotation matrix R
                    - 'intrinsics': (B, T, 3, 3) intrinsic matrix K
                    - 'translation': (B, T, 3) translation vector t
                'attention': list of attention weights (if return_attention=True)
        
        Example:
            >>> output = model(frames, initial_points, return_rays=True, return_camera=True)
            >>> predicted_points = output['points']
            >>> predicted_rays = output['rays']
            >>> camera_center = output['camera']['center']
        """
        B, T, C, H, W = frames.shape
        N = initial_points.shape[1]
        
        # Extract frame features for all frames [0, T]
        frame_features = self.feature_extractor(frames)  # (B, T, C_feat, H_feat, W_feat)
        
        # Track points through frames iteratively with transformer processing
        # At each step, use temporal context and attention from all previous frames
        tracked_points = initial_points.unsqueeze(1)  # (B, 1, N, 2)
        
        # Track points frame by frame with full transformer processing
        for t in range(T - 1):
            
            # Extract features for all tracked frames [0, t]
            # At iteration t, we have t+1 frames tracked (indices 0 through t)
            num_tracked_frames = t + 1
            frame_feat_history = frame_features[:, :num_tracked_frames, :, :, :]  # (B, t+1, C_feat, H_feat, W_feat)
            points_history = tracked_points[:, :num_tracked_frames, :, :]  # (B, t+1, N, 2)
            
            # Extract point features for all tracked frames
            point_features_history = self.extract_point_features(
                frame_feat_history, 
                points_history
            )  # (B, t+1, N, C_feat)
            
            # Get the next frame's features (frame t+1)
            frame_feat_next = frame_features[:, t+1, :, :, :]  # (B, C_feat, H_feat, W_feat)
            frame_feat_current = frame_features[:, t, :, :, :]  # (B, C_feat, H_feat, W_feat)
            
            # Use last tracked points as starting point for next frame
            last_points = tracked_points[:, -1, :, :]  # (B, N, 2)
            
            # Extract features for next frame using correlation-based matching or simple sampling
            if self.use_correlation_matching:
                # Compute correlation volume to handle position uncertainty
                correlation_volume, offset_coords = self.compute_correlation_volume(
                    frame_feat_current,
                    frame_feat_next,
                    last_points,
                    search_radius=self.search_radius
                )  # (B, N, search_area), (B, N, search_area, 2)
                
                # Project correlation volume to feature space
                corr_features = self.corr_proj(correlation_volume)  # (B, N, hidden_dim)
                
                # Also extract point features at predicted location
                frame_feat_next_expanded = frame_feat_next.unsqueeze(1)  # (B, 1, C_feat, H_feat, W_feat)
                last_points_expanded = last_points.unsqueeze(1)  # (B, 1, N, 2)
                point_feat_next = self.extract_point_features(
                    frame_feat_next_expanded, 
                    last_points_expanded
                )  # (B, 1, N, C_feat)
                point_feat_next_flat = point_feat_next.reshape(B, N, self.feature_dim)
            else:
                # Simple feature extraction at predicted location
                frame_feat_next_expanded = frame_feat_next.unsqueeze(1)  # (B, 1, C_feat, H_feat, W_feat)
                last_points_expanded = last_points.unsqueeze(1)  # (B, 1, N, 2)
                point_feat_next = self.extract_point_features(
                    frame_feat_next_expanded, 
                    last_points_expanded
                )  # (B, 1, N, C_feat)
                point_feat_next_flat = point_feat_next.reshape(B, N, self.feature_dim)
            
            # Flatten temporal and point dimensions for history
            point_features_history_flat = point_features_history.reshape(B, num_tracked_frames * N, self.feature_dim)
            
            # Project to hidden dimension
            point_feat_history_proj = self.point_feat_proj(point_features_history_flat)  # (B, (t+1)*N, hidden_dim)
            point_feat_next_proj = self.point_feat_proj(point_feat_next_flat)  # (B, N, hidden_dim)
            
            # Add correlation features if using correlation matching
            if self.use_correlation_matching:
                # Add correlation features to visual features
                # Both are now hidden_dim size, so we can add directly
                point_feat_next_proj = point_feat_next_proj + corr_features  # (B, N, hidden_dim)
            
            # Add point coordinate embeddings
            point_coords_history = points_history.reshape(B, num_tracked_frames * N, 2)
            point_embeds_history = self.point_embed(point_coords_history)  # (B, (t+1)*N, hidden_dim)
            point_embeds_next = self.point_embed(last_points)  # (B, N, hidden_dim)
            
            # Add sinusoidal temporal positional encodings
            temporal_positions_history = torch.arange(num_tracked_frames, device=frames.device).unsqueeze(1).repeat(1, N).reshape(1, -1).repeat(B, 1)
            temporal_embeds_history = self.get_sinusoidal_positional_encoding(
                temporal_positions_history, self.hidden_dim
            )  # (B, (t+1)*N, hidden_dim)
            
            temporal_positions_next = torch.full((B, N), t+1, device=frames.device)
            temporal_embeds_next = self.get_sinusoidal_positional_encoding(
                temporal_positions_next, self.hidden_dim
            )  # (B, N, hidden_dim)
            
            # Combine all features: visual features + point embeddings + temporal embeddings
            x_history = point_feat_history_proj + point_embeds_history + temporal_embeds_history  # (B, (t+1)*N, hidden_dim)
            x_next = point_feat_next_proj + point_embeds_next + temporal_embeds_next  # (B, N, hidden_dim)
            
            # Reshape to separate frames: (B, num_frames, N, hidden_dim)
            x_history_frames = x_history.reshape(B, num_tracked_frames, N, self.hidden_dim)
            x_next_frame = x_next.unsqueeze(1)  # (B, 1, N, hidden_dim)
            
            # Create temporal tokens for each frame
            temporal_tokens_history = self.temporal_token.expand(B, num_tracked_frames, -1)  # (B, t+1, hidden_dim)
            temporal_token_next = self.temporal_token.expand(B, 1, -1)  # (B, 1, hidden_dim)
            
            # Add temporal positional encoding to temporal tokens
            temporal_pos_tokens_history = torch.arange(num_tracked_frames, device=frames.device).unsqueeze(0).repeat(B, 1)  # (B, t+1)
            temporal_embed_tokens_history = self.get_sinusoidal_positional_encoding(
                temporal_pos_tokens_history, self.hidden_dim
            )  # (B, t+1, hidden_dim)
            temporal_tokens_history = temporal_tokens_history + temporal_embed_tokens_history
            
            temporal_pos_token_next = torch.full((B, 1), t+1, dtype=torch.long, device=frames.device)  # (B, 1)
            temporal_embed_token_next = self.get_sinusoidal_positional_encoding(
                temporal_pos_token_next, self.hidden_dim
            )  # (B, 1, hidden_dim)
            temporal_token_next = temporal_token_next + temporal_embed_token_next
            
            # Interleave temporal tokens with point tokens
            # For each frame: [temporal_token, point1, point2, ..., pointN]
            x_history_with_tokens = []
            for i in range(num_tracked_frames):
                x_history_with_tokens.append(temporal_tokens_history[:, i:i+1, :])  # Temporal token
                x_history_with_tokens.append(x_history_frames[:, i, :, :])  # N point tokens
            x_history_seq = torch.cat(x_history_with_tokens, dim=1)  # (B, (t+1)*(N+1), hidden_dim)
            
            x_next_with_token = torch.cat([temporal_token_next, x_next_frame.squeeze(1)], dim=1)  # (B, N+1, hidden_dim)
            
            # Concatenate history and next frame features
            x = torch.cat([x_history_seq, x_next_with_token], dim=1)  # (B, (t+2)*(N+1), hidden_dim)
            
            # Process through transformer layers with attention
            for layer in self.transformer_layers:
                x, _ = layer(x)
            
            # Extract features for the next frame (last N tokens = point tokens only)
            x_next_out = x[:, -N:, :]  # (B, N, hidden_dim) - last N tokens are the point tokens
            
            # Predict offset for next frame using tracking head
            offset = self.tracking_head(x_next_out)  # (B, N, 2)
            next_points = last_points + offset
            
            # Append to tracked points
            tracked_points = torch.cat([tracked_points, next_points.unsqueeze(1)], dim=1)  # (B, t+2, N, 2)
        
        # Now we have tracked points for frames [0, T-1] (T frames total)
        # tracked_points: (B, T, N, 2)
        
        # For final prediction, extract features for all frames including the last one
        # Extract point features for all tracked frames [0, T-1]
        point_features_all = self.extract_point_features(
            frame_features, 
            tracked_points
        )  # (B, T, N, C_feat)
        
        # Flatten temporal and point dimensions
        point_features_all_flat = point_features_all.reshape(B, T * N, self.feature_dim)  # (B, T*N, C_feat)
        
        # Project to hidden dimension
        point_feat_all_proj = self.point_feat_proj(point_features_all_flat)  # (B, T*N, hidden_dim)
        
        # Add point coordinate embeddings
        point_coords_all = tracked_points.reshape(B, T * N, 2)  # (B, T*N, 2)
        point_embeds_all = self.point_embed(point_coords_all)  # (B, T*N, hidden_dim)
        
        # Add sinusoidal temporal positional encodings
        temporal_positions_all = torch.arange(T, device=frames.device).unsqueeze(1).repeat(1, N).reshape(1, -1).repeat(B, 1)  # (B, T*N)
        temporal_embeds_all = self.get_sinusoidal_positional_encoding(
            temporal_positions_all, self.hidden_dim
        )  # (B, T*N, hidden_dim)
        
        # Combine all features
        x_all = point_feat_all_proj + point_embeds_all + temporal_embeds_all  # (B, T*N, hidden_dim)
        
        # Reshape to separate frames
        x_all_frames = x_all.reshape(B, T, N, self.hidden_dim)  # (B, T, N, hidden_dim)
        
        # Create temporal tokens for all frames
        temporal_tokens_all = self.temporal_token.expand(B, T, -1)  # (B, T, hidden_dim)
        
        # Add temporal positional encoding to temporal tokens
        temporal_pos_tokens_all = torch.arange(T, device=frames.device).unsqueeze(0).repeat(B, 1)  # (B, T)
        temporal_embed_tokens_all = self.get_sinusoidal_positional_encoding(
            temporal_pos_tokens_all, self.hidden_dim
        )  # (B, T, hidden_dim)
        temporal_tokens_all = temporal_tokens_all + temporal_embed_tokens_all
        
        # Interleave temporal tokens with point tokens
        # For each frame: [temporal_token, point1, point2, ..., pointN]
        x_with_tokens = []
        for i in range(T):
            x_with_tokens.append(temporal_tokens_all[:, i:i+1, :])  # Temporal token
            x_with_tokens.append(x_all_frames[:, i, :, :])  # N point tokens
        x = torch.cat(x_with_tokens, dim=1)  # (B, T*(N+1), hidden_dim)
        
        # Final refinement pass through transformer with full temporal context
        attention_weights_list = []
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)
            attention_weights_list.append(attn_weights)
        
        # Extract only point tokens (skip temporal tokens) and reshape
        # Sequence is: [temp0, pt0_0, ..., pt0_N-1, temp1, pt1_0, ..., pt1_N-1, ...]
        x_points_only = []
        for i in range(T):
            start_idx = i * (N + 1) + 1  # Skip temporal token
            end_idx = start_idx + N
            x_points_only.append(x[:, start_idx:end_idx, :])  # (B, N, hidden_dim)
        x_points = torch.stack(x_points_only, dim=1)  # (B, T, N, hidden_dim)
        
        # Refine all tracked points using the output head
        x_flat_for_pred = x_points.reshape(B * T, N, self.hidden_dim)  # (B*T, N, hidden_dim)
        
        # Predict offsets for all frames
        offsets_all = self.output_head(x_flat_for_pred)  # (B*T, N, 2)
        offsets_all = offsets_all.reshape(B, T, N, 2)  # (B, T, N, 2)
        
        # Add offsets to tracked points to get refined predictions
        predicted_points = tracked_points + offsets_all  # (B, T, N, 2)
        
        # Predict rays for all points across all frames (Plücker coordinates)
        predicted_rays = None
        camera_params = None
        
        if return_rays or return_camera:
            # Use only point features (not temporal tokens) for ray prediction
            x_points_flat = x_points.reshape(B * T * N, self.hidden_dim)  # (B*T*N, hidden_dim)
            ray_plucker = self.ray_head(x_points_flat)  # (B*T*N, 6) - Plücker coords [d, m]
            predicted_rays = ray_plucker.reshape(B, T, N, 6)  # (B, T, N, 6)
            
            # Normalize ray directions
            ray_directions = predicted_rays[..., :3]  # (B, T, N, 3)
            ray_moments = predicted_rays[..., 3:]     # (B, T, N, 3)
            ray_directions = self.normalize_ray_direction(ray_directions)
            predicted_rays = torch.cat([ray_directions, ray_moments], dim=-1)
        
        if return_camera:
            # Recover camera parameters from predicted rays
            # Use rays from all frames and points to estimate camera parameters per frame
            camera_centers = []
            camera_rotations = []
            camera_intrinsics = []
            camera_translations = []
            
            H, W = frames.shape[-2:]
            
            for t in range(T):
                # Get rays and pixel coordinates for frame t
                rays_t = predicted_rays[:, t, :, :]  # (B, N, 6)
                points_t = predicted_points[:, t, :, :]  # (B, N, 2)
                
                ray_dirs_t = rays_t[..., :3]  # (B, N, 3)
                ray_moms_t = rays_t[..., 3:]  # (B, N, 3)
                
                # Recover camera center (Equation 4)
                center_t = self.recover_camera_center_from_rays(ray_dirs_t, ray_moms_t)  # (B, 3)
                
                # Recover rotation and intrinsics (Equation 5)
                rotation_t, intrinsics_t = self.recover_rotation_intrinsics_from_rays(
                    ray_dirs_t, points_t, image_size=(H, W)
                )  # (B, 3, 3), (B, 3, 3)
                
                # Compute translation: t = -R^T * c
                translation_t = self.compute_translation_from_rotation_center(rotation_t, center_t)  # (B, 3)
                
                camera_centers.append(center_t)
                camera_rotations.append(rotation_t)
                camera_intrinsics.append(intrinsics_t)
                camera_translations.append(translation_t)
            
            # Stack across time
            camera_params = {
                'center': torch.stack(camera_centers, dim=1),      # (B, T, 3)
                'rotation': torch.stack(camera_rotations, dim=1),  # (B, T, 3, 3)
                'intrinsics': torch.stack(camera_intrinsics, dim=1),  # (B, T, 3, 3)
                'translation': torch.stack(camera_translations, dim=1)  # (B, T, 3)
            }
        
        # Return results as dictionary for cleaner API
        results = {
            'points': predicted_points,  # Always returned: (B, T, N, 2)
        }
        
        if return_rays:
            results['rays'] = predicted_rays  # (B, T, N, 6)
        
        if return_camera:
            results['camera'] = camera_params  # dict with center, rotation, intrinsics, translation
        
        if return_attention:
            results['attention'] = attention_weights_list  # list of attention weights
        
        return results
