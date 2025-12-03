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
        max_frames_per_video=None,
    ):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.num_points = num_points
        self.image_size = image_size
        self.min_frames = min_frames
        self.max_frames_per_video = max_frames_per_video
        
        if video_dirs is None:
            self.video_dirs = [d.name for d in self.data_root.iterdir() if d.is_dir()]
        else:
            self.video_dirs = video_dirs
        
        self.video_annotations = {}
        self.sequences = []
        for video_dir_name in self.video_dirs:
            video_dir = self.data_root / video_dir_name
            frames_dir = video_dir / "frames"
            anno_path = video_dir / "anno.npz"
            
            if not frames_dir.exists() or not anno_path.exists():
                print(f"Warning: Skipping {video_dir_name} - missing frames or anno.npz")
                continue
            try:
                with np.load(anno_path) as anno:
                    trajs_2d = anno['trajs_2d'].copy()  # (T, N, 2)
                    visibs = anno['visibs'].copy()  # (T, N)
                    
                    # Load 3D data and camera parameters if available
                    trajs_3d = anno.get('trajs_3d', None)
                    if trajs_3d is not None:
                        trajs_3d = trajs_3d.copy()  # (T, N, 3)
                    
                    intrinsics = anno.get('intrinsics', None)
                    if intrinsics is not None:
                        intrinsics = intrinsics.copy()  # (T, 3, 3) or (3, 3)
                    
                    extrinsics = anno.get('extrinsics', None)
                    if extrinsics is not None:
                        extrinsics = extrinsics.copy()  # (T, 4, 4) or (4, 4)
                
                self.video_annotations[video_dir_name] = {
                    'trajs_2d': trajs_2d,
                    'visibs': visibs,
                    'trajs_3d': trajs_3d,
                    'intrinsics': intrinsics,
                    'extrinsics': extrinsics
                }
            except Exception as e:
                print(f"Warning: Could not load {anno_path}: {e}")
                continue
            
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            if len(frame_files) == 0:
                frame_files = sorted(frames_dir.glob("frame_*.png"))
            
            frames_in_dir = len(frame_files)
            frames_in_anno = trajs_2d.shape[0]
            num_frames = min(frames_in_dir, frames_in_anno)
            
            # Apply max_frames_per_video constraint if specified
            if self.max_frames_per_video is not None:
                num_frames = min(num_frames, self.max_frames_per_video)
                print(f"Info: Limiting {video_dir_name} to {num_frames} frames (max_frames_per_video={self.max_frames_per_video})")
            
            if num_frames < self.min_frames:
                print(f"Warning: Skipping {video_dir_name} - only {num_frames} frames available (minimum: {self.min_frames})")
                continue
            
            if num_frames < sequence_length:
                print(f"Warning: Skipping {video_dir_name} - only {num_frames} frames available (need at least {sequence_length} for sequence_length)")
                continue
            
            for start_idx in range(num_frames - sequence_length + 1):
                start_frame_visibs = visibs[start_idx]
                valid_point_indices = [
                    point_idx for point_idx in range(visibs.shape[1])
                    if start_frame_visibs[point_idx]
                ]
                
                if len(valid_point_indices) == 0:
                    continue
                
                self.sequences.append({
                    'video_dir': video_dir_name,
                    'start_idx': start_idx,
                    'valid_indices': valid_point_indices
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def load_frame(self, video_dir_name, frame_idx):
        """Load and preprocess a single frame"""
        video_dir = self.data_root / video_dir_name
        frames_dir = video_dir / "frames"
        
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
        valid_indices = seq_info['valid_indices']
        num_selectable = min(self.num_points, len(valid_indices))
        selected_indices = random.sample(valid_indices, num_selectable)
        
        anno_data = self.video_annotations[video_dir_name]
        trajs_2d = anno_data['trajs_2d']
        visibs = anno_data['visibs']
        trajs_3d = anno_data.get('trajs_3d', None)
        intrinsics = anno_data.get('intrinsics', None)
        extrinsics = anno_data.get('extrinsics', None)
        
        frame_indices = list(range(start_idx, start_idx + self.sequence_length))
        
        frames = []
        original_sizes = []
        for frame_idx in frame_indices:
            frame, original_size = self.load_frame(video_dir_name, frame_idx)
            frames.append(frame)
            original_sizes.append(original_size)
        
        frames = np.stack(frames)  # (T, C, H, W)
        
        # trajs_2d: (T_anno, N_total, 2)
        # visibs: (T_anno, N_total)
        # trajs_3d: (T_anno, N_total, 3) if available
        gt_trajectories = []
        gt_visibilities = []
        gt_trajectories_3d = []
        initial_points = None
        
        # Collect intrinsics and extrinsics for selected frames
        gt_intrinsics = []
        gt_extrinsics = []
        
        for t, frame_idx in enumerate(frame_indices):
            if frame_idx >= trajs_2d.shape[0]:
                frame_idx = trajs_2d.shape[0] - 1
            
            frame_coords = trajs_2d[frame_idx, selected_indices, :]  # (N, 2)
            frame_visibs = visibs[frame_idx, selected_indices]  # (N,)
            
            original_size = original_sizes[t]
            normalized_coords = frame_coords.copy()
            normalized_coords[:, 0] /= original_size[0]
            normalized_coords[:, 1] /= original_size[1]
            
            if t == 0:
                initial_points = normalized_coords.copy()
            
            gt_trajectories.append(normalized_coords)
            gt_visibilities.append(frame_visibs)
            
            # Collect 3D trajectories if available
            if trajs_3d is not None:
                frame_3d = trajs_3d[frame_idx, selected_indices, :]  # (N, 3)
                gt_trajectories_3d.append(frame_3d)
            
            # Collect intrinsics
            if intrinsics is not None:
                if len(intrinsics.shape) == 3:
                    # Intrinsics vary per frame: (T, 3, 3)
                    gt_intrinsics.append(intrinsics[frame_idx])
                else:
                    # Constant intrinsics: (3, 3)
                    gt_intrinsics.append(intrinsics)
            
            # Collect extrinsics
            if extrinsics is not None:
                if len(extrinsics.shape) == 3:
                    # Extrinsics vary per frame: (T, 4, 4)
                    gt_extrinsics.append(extrinsics[frame_idx])
                else:
                    # Constant extrinsics: (4, 4)
                    gt_extrinsics.append(extrinsics)
        
        gt_trajectories = np.array(gt_trajectories, dtype=np.float32)  # (T, N, 2)
        gt_visibilities = np.array(gt_visibilities, dtype=np.float32)  # (T, N)
        initial_points = np.array(initial_points, dtype=np.float32)  # (N, 2)
        
        # Convert to torch tensors
        frames = torch.from_numpy(frames).float()
        initial_points = torch.from_numpy(initial_points).float()
        gt_trajectories = torch.from_numpy(gt_trajectories).float()
        gt_visibilities = torch.from_numpy(gt_visibilities).float()
        
        # Prepare return dictionary
        result = {
            'frames': frames,  # (T, C, H, W)
            'initial_points': initial_points,  # (N, 2)
            'gt_trajectories': gt_trajectories,  # (T, N, 2)
            'gt_visibilities': gt_visibilities,  # (T, N)
            'video_name': video_dir_name,
            'start_idx': start_idx
        }
        
        # Add 3D data if available
        if len(gt_trajectories_3d) > 0:
            gt_trajectories_3d = np.array(gt_trajectories_3d, dtype=np.float32)  # (T, N, 3)
            result['gt_trajectories_3d'] = torch.from_numpy(gt_trajectories_3d).float()
        
        if len(gt_intrinsics) > 0:
            gt_intrinsics = np.array(gt_intrinsics, dtype=np.float32)  # (T, 3, 3)
            result['gt_intrinsics'] = torch.from_numpy(gt_intrinsics).float()
        
        if len(gt_extrinsics) > 0:
            gt_extrinsics = np.array(gt_extrinsics, dtype=np.float32)  # (T, 4, 4)
            result['gt_extrinsics'] = torch.from_numpy(gt_extrinsics).float()
        
        return result
