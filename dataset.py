"""
Data Loader for Point Odyssey Dataset
Matches the loading approach from "Point Tracking With Intrinsics v2"
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import random


def normalize_coordinates(coordinates, image_size):
    """Normalize coordinates to [0, 1] range"""
    width, height = image_size
    normalized_coordinates = coordinates.clone()
    normalized_coordinates[:, 0] /= width
    normalized_coordinates[:, 1] /= height
    return normalized_coordinates


def get_valid_point_indices(anno):
    """
    Get valid point indices from annotation data.
    Points are valid if they are visible in the first frame.
    
    Args:
        anno: Annotation data loaded from npz file
        
    Returns:
        List of valid point indices
    """
    total_points = anno['trajs_2d'].shape[1]
    valid_point_indices = []
    first_frame_visibs = anno['visibs'][0]
    
    for point_idx in range(total_points):
        if first_frame_visibs[point_idx]:
            valid_point_indices.append(point_idx)
    
    return valid_point_indices


class PointOdysseyDataset(Dataset):
    """
    Dataset loader for Point Odyssey dataset
    Matches the data structure from "Point Tracking With Intrinsics v2"
    
    Assumes data structure:
    data_dir/
        video_dir/
            frames/
                frame_000000.jpg
                frame_000001.jpg
                ...
            anno.npz  # Contains trajs_2d and visibs
    """
    
    def __init__(
        self,
        data_root,
        video_dirs=None,
        sequence_length=8,
        num_points=8,
        image_size=(256, 256),
        min_frames=20,
        augment=True
    ):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.num_points = num_points
        self.image_size = image_size
        self.min_frames = min_frames
        self.augment = augment
        
        # Get video directories
        if video_dirs is None:
            # Find all video directories
            self.video_dirs = [d.name for d in self.data_root.iterdir() if d.is_dir()]
        else:
            self.video_dirs = video_dirs
        
        # Build list of valid sequences
        self.sequences = []
        for video_dir_name in self.video_dirs:
            video_dir = self.data_root / video_dir_name
            frames_dir = video_dir / "frames"
            anno_path = video_dir / "anno.npz"
            
            if not frames_dir.exists() or not anno_path.exists():
                print(f"Warning: Skipping {video_dir_name} - missing frames or anno.npz")
                continue
            
            # Load annotation
            try:
                anno = np.load(anno_path)
                trajs_2d = anno['trajs_2d']  # (T, N, 2)
                visibs = anno['visibs']  # (T, N)
            except Exception as e:
                print(f"Warning: Could not load {anno_path}: {e}")
                continue
            
            # Get frame files
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            if len(frame_files) == 0:
                frame_files = sorted(frames_dir.glob("frame_*.png"))
            
            frames_in_dir = len(frame_files)
            frames_in_anno = trajs_2d.shape[0]
            num_frames = min(self.min_frames, min(frames_in_dir, frames_in_anno))
            
            if num_frames < sequence_length:
                print(f"Warning: Skipping {video_dir_name} - only {num_frames} frames available")
                continue
            
            # Get valid point indices (visible in first frame)
            valid_point_indices = get_valid_point_indices(anno)
            
            if len(valid_point_indices) == 0:
                print(f"Warning: Skipping {video_dir_name} - no valid points in first frame")
                continue
            
            # Create sequences
            for start_idx in range(num_frames - sequence_length + 1):
                # Sample points for this sequence
                selected_indices = random.sample(
                    valid_point_indices, 
                    min(self.num_points, len(valid_point_indices))
                )
                
                self.sequences.append({
                    'video_dir': video_dir_name,
                    'start_idx': start_idx,
                    'anno': anno,
                    'selected_indices': selected_indices
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def load_frame(self, video_dir_name, frame_idx):
        """Load and preprocess a single frame"""
        video_dir = self.data_root / video_dir_name
        frames_dir = video_dir / "frames"
        
        # Try jpg first, then png
        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
        if not frame_path.exists():
            frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
        
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {frame_path}")
        
        img = Image.open(frame_path).convert('RGB')
        original_size = img.size  # (width, height)
        img = img.resize(self.image_size)
        img = np.array(img).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        return img, original_size
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        video_dir_name = seq_info['video_dir']
        start_idx = seq_info['start_idx']
        anno = seq_info['anno']
        selected_indices = seq_info['selected_indices']
        
        # Get frame indices for this sequence
        frame_indices = list(range(start_idx, start_idx + self.sequence_length))
        
        # Load frames
        frames = []
        original_sizes = []
        for frame_idx in frame_indices:
            frame, original_size = self.load_frame(video_dir_name, frame_idx)
            frames.append(frame)
            original_sizes.append(original_size)
        
        frames = np.stack(frames)  # (T, C, H, W)
        
        # Get ground truth trajectories from annotation
        # trajs_2d: (T_anno, N_total, 2) in pixel coordinates
        # visibs: (T_anno, N_total) boolean visibility
        trajs_2d = anno['trajs_2d']
        visibs = anno['visibs']
        
        # Extract trajectories for selected points
        gt_trajectories = []
        gt_visibilities = []
        initial_points = None
        
        for t, frame_idx in enumerate(frame_indices):
            if frame_idx >= trajs_2d.shape[0]:
                # Pad with last frame if needed
                frame_idx = trajs_2d.shape[0] - 1
            
            # Get coordinates for selected points at this frame
            frame_coords = trajs_2d[frame_idx, selected_indices, :]  # (N, 2)
            frame_visibs = visibs[frame_idx, selected_indices]  # (N,)
            
            # Normalize coordinates based on original image size
            original_size = original_sizes[t]
            normalized_coords = frame_coords.copy()
            normalized_coords[:, 0] /= original_size[0]  # width
            normalized_coords[:, 1] /= original_size[1]  # height
            
            # Store first frame as initial points
            if t == 0:
                initial_points = normalized_coords.copy()
            
            gt_trajectories.append(normalized_coords)
            gt_visibilities.append(frame_visibs)
        
        # Pad if we have fewer points than num_points
        num_selected = len(selected_indices)
        if num_selected < self.num_points:
            # Pad with zeros (or last point)
            pad_size = self.num_points - num_selected
            for t in range(len(gt_trajectories)):
                # Pad coordinates with center point
                pad_coords = np.ones((pad_size, 2), dtype=np.float32) * 0.5
                gt_trajectories[t] = np.vstack([gt_trajectories[t], pad_coords])
                # Pad visibilities with False
                pad_visibs = np.zeros(pad_size, dtype=np.bool_)
                gt_visibilities[t] = np.hstack([gt_visibilities[t], pad_visibs])
            
            # Pad initial points
            pad_coords = np.ones((pad_size, 2), dtype=np.float32) * 0.5
            initial_points = np.vstack([initial_points, pad_coords])
        
        # Convert to numpy arrays
        gt_trajectories = np.array(gt_trajectories, dtype=np.float32)  # (T, N, 2)
        gt_visibilities = np.array(gt_visibilities, dtype=np.float32)  # (T, N)
        initial_points = np.array(initial_points, dtype=np.float32)  # (N, 2)
        
        # Convert to torch tensors
        frames = torch.from_numpy(frames).float()
        initial_points = torch.from_numpy(initial_points).float()
        gt_trajectories = torch.from_numpy(gt_trajectories).float()
        gt_visibilities = torch.from_numpy(gt_visibilities).float()
        
        return {
            'frames': frames,  # (T, C, H, W)
            'initial_points': initial_points,  # (N, 2)
            'gt_trajectories': gt_trajectories,  # (T, N, 2)
            'gt_visibilities': gt_visibilities,  # (T, N)
            'video_name': video_dir_name,
            'start_idx': start_idx
        }


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing when Point Odyssey data is not available
    Creates simple moving points on synthetic backgrounds
    """
    
    def __init__(
        self,
        num_samples=1000,
        sequence_length=8,
        num_points=8,
        image_size=(256, 256),
        augment=True
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_points = num_points
        self.image_size = image_size
        self.augment = augment
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        H, W = self.image_size
        
        # Generate random background
        frames = []
        gt_trajectories = []
        
        # Initialize random points
        initial_points = np.random.rand(self.num_points, 2).astype(np.float32)
        
        # Generate trajectories with random motion
        current_points = initial_points.copy()
        for t in range(self.sequence_length):
            # Create frame with random background
            frame = np.random.rand(3, H, W).astype(np.float32) * 0.3 + 0.2
            
            # Add some structure to background
            y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            pattern = np.sin(x_coords * 0.1) * np.cos(y_coords * 0.1) * 0.1
            frame += pattern[np.newaxis, :, :]
            
            # Add point markers (small circles)
            for point in current_points:
                px, py = int(point[0] * W), int(point[1] * H)
                px, py = np.clip(px, 0, W-1), np.clip(py, 0, H-1)
                
                # Draw small circle
                y, x = np.ogrid[:H, :W]
                mask = (x - px)**2 + (y - py)**2 <= 25  # radius 5
                frame[:, mask] = 1.0  # White point
            
            frames.append(frame)
            gt_trajectories.append(current_points.copy())
            
            # Update points with random motion
            if t < self.sequence_length - 1:
                velocity = np.random.randn(self.num_points, 2) * 0.02
                current_points += velocity
                current_points = np.clip(current_points, 0.1, 0.9)
        
        frames = np.stack(frames)  # (T, C, H, W)
        gt_trajectories = np.stack(gt_trajectories)  # (T, N, 2)
        
        # Convert to torch
        frames = torch.from_numpy(frames).float()
        initial_points = torch.from_numpy(initial_points).float()
        gt_trajectories = torch.from_numpy(gt_trajectories).float()
        
        return {
            'frames': frames,
            'initial_points': initial_points,
            'gt_trajectories': gt_trajectories,
            'video_name': f'synthetic_{idx}',
            'start_idx': 0
        }

