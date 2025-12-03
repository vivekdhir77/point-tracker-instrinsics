"""
Training script for Point Tracker with visualization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

from model import PointTracker
from dataset import PointOdysseyDataset
from visualize import visualize_attention, plot_loss_history, visualize_predictions


def extract_camera_parameters_from_extrinsics(extrinsics):
    """
    Extract camera parameters from extrinsic matrix
    
    Args:
        extrinsics: (B, T, 4, 4) or (T, 4, 4) extrinsic matrices
        
    Returns:
        dict with:
            'center': camera center in world coordinates
            'rotation': rotation matrix R
            'translation': translation vector t
    """
    is_batched = len(extrinsics.shape) == 4
    
    if is_batched:
        # (B, T, 4, 4)
        R = extrinsics[:, :, :3, :3]  # (B, T, 3, 3)
        t = extrinsics[:, :, :3, 3]   # (B, T, 3)
    else:
        # (T, 4, 4)
        R = extrinsics[:, :3, :3]  # (T, 3, 3)
        t = extrinsics[:, :3, 3]   # (T, 3)
    
    # Camera center: C = -R^T @ t
    # For batched: need to handle batch matrix multiplication
    if is_batched:
        B, T = R.shape[:2]
        R_T = R.transpose(-2, -1)  # (B, T, 3, 3)
        C = -torch.bmm(R_T.reshape(B*T, 3, 3), t.reshape(B*T, 3, 1)).reshape(B, T, 3)
    else:
        T = R.shape[0]
        R_T = R.transpose(-2, -1)  # (T, 3, 3)
        C = -torch.bmm(R_T, t.unsqueeze(-1)).squeeze(-1)  # (T, 3)
    
    return {
        'center': C,
        'rotation': R,
        'translation': t
    }


def compute_ground_truth_rays(points_3d_world, camera_center):
    """
    Compute ground truth rays from 3D points and camera center
    
    Args:
        points_3d_world: (B, T, N, 3) 3D points in world coordinates
        camera_center: (B, T, 3) camera center in world coordinates
        
    Returns:
        rays: (B, T, N, 6) rays in Plücker coordinates [d_x, d_y, d_z, m_x, m_y, m_z]
    """
    # Expand camera center to match points shape
    camera_center_expanded = camera_center.unsqueeze(2)  # (B, T, 1, 3)
    
    # Ray direction: from camera center to 3D point
    ray_directions = points_3d_world - camera_center_expanded  # (B, T, N, 3)
    
    # Normalize directions
    ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1, eps=1e-6)
    
    # Moment: m = c × d (camera center cross ray direction)
    # Cross product for batched tensors
    ray_moments = torch.cross(
        camera_center_expanded.expand_as(ray_directions), 
        ray_directions, 
        dim=-1
    )  # (B, T, N, 3)
    
    # Concatenate to form Plücker coordinates
    rays = torch.cat([ray_directions, ray_moments], dim=-1)  # (B, T, N, 6)
    
    return rays


def compute_loss(pred_points, gt_points, initial_points, pred_rays=None, gt_rays=None, 
                 pred_visibility=None, gt_visibility=None, lambda_ray=0.8, lambda_visibility=0.5):
    """
    Compute tracking loss, ray prediction loss, and visibility loss
    
    Args:
        pred_points: (B, T, N, 2) predicted points
        gt_points: (B, T, N, 2) ground truth points
        initial_points: (B, N, 2) initial points
        pred_rays: (B, T, N, 6) predicted rays in Plücker coordinates (optional)
        gt_rays: (B, T, N, 6) ground truth rays (optional)
        pred_visibility: (B, T, N) predicted visibility probabilities (optional)
        gt_visibility: (B, T, N) ground truth visibility (0 or 1) (optional)
        lambda_ray: weight for ray prediction loss
        lambda_visibility: weight for visibility loss
    
    Returns:
        dict with loss tensors (using sum of absolute errors, not mean)
    
    Note:
        Ray loss implicitly supervises camera parameters through geometric constraints.
        If rays are correct, camera recovery will be correct too - no need for separate camera loss!
    """
    # 2D tracking losses - use SUM of absolute errors
    point_error = torch.abs(pred_points - gt_points)  # (B, T, N, 2)
    l1_loss = point_error.sum()
    
    # End-point error: L2 distance per point
    epe = torch.sqrt(torch.sum(point_error ** 2, dim=-1))  # (B, T, N)
    epe_loss = epe.sum()
    
    # Temporal consistency loss
    if pred_points.shape[1] > 1:
        pred_velocities = pred_points[:, 1:] - pred_points[:, :-1]  # (B, T-1, N, 2)
        gt_velocities = gt_points[:, 1:] - gt_points[:, :-1]
        velocity_error = torch.abs(pred_velocities - gt_velocities)
        temporal_loss = velocity_error.sum()
    else:
        temporal_loss = torch.tensor(0.0, device=pred_points.device)
    
    # Ray prediction loss - implicitly supervises camera parameters!
    ray_loss = torch.tensor(0.0, device=pred_points.device)
    direction_loss = torch.tensor(0.0, device=pred_points.device)
    moment_loss = torch.tensor(0.0, device=pred_points.device)
    
    if pred_rays is not None and gt_rays is not None:
        # Separate direction and moment components
        pred_dirs = pred_rays[..., :3]  # (B, T, N, 3)
        gt_dirs = gt_rays[..., :3]
        pred_moments = pred_rays[..., 3:]  # (B, T, N, 3)
        gt_moments = gt_rays[..., 3:]
        
        # Direction loss: Angular error (1 - cosine similarity)
        # Better for unit vectors than L2 distance
        cos_sim = torch.nn.functional.cosine_similarity(
            pred_dirs.reshape(-1, 3), 
            gt_dirs.reshape(-1, 3), 
            dim=-1
        )  # (B*T*N,)
        direction_loss = (1 - cos_sim).sum()  # Sum, not mean
        
        # Moment loss: Sum of absolute errors
        moment_error = torch.abs(pred_moments - gt_moments)  # (B, T, N, 3)
        moment_loss = moment_error.sum()
        
        # Combined ray loss
        ray_loss = direction_loss + moment_loss
    
    # Visibility loss - binary cross-entropy
    visibility_loss = torch.tensor(0.0, device=pred_points.device)
    if pred_visibility is not None and gt_visibility is not None:
        # Binary cross-entropy loss
        # pred_visibility: (B, T, N) probabilities [0, 1]
        # gt_visibility: (B, T, N) binary values [0, 1]
        visibility_loss = torch.nn.functional.binary_cross_entropy(
            pred_visibility, 
            gt_visibility, 
            reduction='sum'  # Sum, not mean (consistent with other losses)
        )
    
    # Total loss (no camera loss - it's redundant with ray loss!)
    total_loss = l1_loss + 0.1 * temporal_loss + lambda_ray * ray_loss + lambda_visibility * visibility_loss
    
    return {
        'total_loss': total_loss,
        'l1_loss': l1_loss,
        'epe': epe_loss,
        'temporal_loss': temporal_loss,
        'ray_loss': ray_loss,
        'direction_loss': direction_loss,
        'moment_loss': moment_loss,
        'visibility_loss': visibility_loss,
    }


def accumulate_losses(total_losses, losses, num_batches):
    """Helper function to accumulate losses"""
    for key in total_losses:
        total_losses[key] += losses[key].item()
    return num_batches + 1


def plot_detailed_losses(epoch_losses_history, batch_losses_history, save_path):
    """
    Plot detailed loss curves including per-epoch and per-batch losses
    
    Args:
        epoch_losses_history: dict with 'train' and 'val' keys, each containing dicts of loss lists
        batch_losses_history: dict with loss names as keys, values are lists of per-batch losses
        save_path: path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss (Train vs Val per epoch)
    ax = axes[0, 0]
    if 'train' in epoch_losses_history and len(epoch_losses_history['train']) > 0:
        epochs = range(len(epoch_losses_history['train']))
        train_losses = [losses_dict['total_loss'] for losses_dict in epoch_losses_history['train']]
        val_losses = [losses_dict['total_loss'] for losses_dict in epoch_losses_history['val']]
        ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss per Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: EPE (End-Point Error)
    ax = axes[0, 1]
    if 'train' in epoch_losses_history and len(epoch_losses_history['train']) > 0:
        epochs = range(len(epoch_losses_history['train']))
        train_epe = [losses_dict['epe'] for losses_dict in epoch_losses_history['train']]
        val_epe = [losses_dict['epe'] for losses_dict in epoch_losses_history['val']]
        ax.plot(epochs, train_epe, 'b-', label='Train', linewidth=2)
        ax.plot(epochs, val_epe, 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('EPE')
        ax.set_title('End-Point Error per Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Ray Loss Components
    ax = axes[0, 2]
    if 'train' in epoch_losses_history and len(epoch_losses_history['train']) > 0:
        epochs = range(len(epoch_losses_history['train']))
        if 'ray_loss' in epoch_losses_history['train'][0]:
            train_ray = [losses_dict['ray_loss'] for losses_dict in epoch_losses_history['train']]
            val_ray = [losses_dict['ray_loss'] for losses_dict in epoch_losses_history['val']]
            ax.plot(epochs, train_ray, 'b-', label='Train Ray', linewidth=2)
            ax.plot(epochs, val_ray, 'r-', label='Val Ray', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Ray Loss')
            ax.set_title('Ray Loss per Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Ray loss not enabled', ha='center', va='center')
            ax.set_title('Ray Loss (Not Enabled)')
    
    # Plot 4: Per-Batch Total Loss (shows training dynamics)
    ax = axes[1, 0]
    if 'total_loss' in batch_losses_history and len(batch_losses_history['total_loss']) > 0:
        batches = range(len(batch_losses_history['total_loss']))
        ax.plot(batches, batch_losses_history['total_loss'], 'b-', alpha=0.6, linewidth=0.5)
        # Add moving average
        window = min(50, len(batch_losses_history['total_loss']) // 10)
        if window > 1:
            moving_avg = np.convolve(batch_losses_history['total_loss'], 
                                    np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(batch_losses_history['total_loss'])), 
                   moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
            ax.legend()
        ax.set_xlabel('Batch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Per-Batch Total Loss')
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Direction vs Moment Loss
    ax = axes[1, 1]
    if 'direction_loss' in batch_losses_history and len(batch_losses_history['direction_loss']) > 0:
        batches = range(len(batch_losses_history['direction_loss']))
        ax.plot(batches, batch_losses_history['direction_loss'], 'g-', alpha=0.4, linewidth=0.5, label='Direction')
        ax.plot(batches, batch_losses_history['moment_loss'], 'm-', alpha=0.4, linewidth=0.5, label='Moment')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Ray Components (Per-Batch)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Ray components not available', ha='center', va='center')
        ax.set_title('Ray Components (Not Enabled)')
    
    # Plot 6: L1 vs Temporal Loss
    ax = axes[1, 2]
    if 'train' in epoch_losses_history and len(epoch_losses_history['train']) > 0:
        epochs = range(len(epoch_losses_history['train']))
        train_l1 = [losses_dict['l1_loss'] for losses_dict in epoch_losses_history['train']]
        train_temporal = [losses_dict['temporal_loss'] for losses_dict in epoch_losses_history['train']]
        ax.plot(epochs, train_l1, 'b-', label='L1', linewidth=2)
        ax.plot(epochs, train_temporal, 'g-', label='Temporal', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components per Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved detailed loss plot: {save_path}')


def train_epoch(model, dataloader, optimizer, device, epoch, save_dir, batch_losses_history,
                use_rays=False, lambda_ray=0.1):
    """Train for one epoch"""
    model.train()
    total_losses = {
        'total_loss': 0, 'l1_loss': 0, 'epe': 0, 'temporal_loss': 0,
        'ray_loss': 0, 'direction_loss': 0, 'moment_loss': 0,
        'visibility_loss': 0
    }
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        initial_points = batch['initial_points'].to(device)  # (B, N, 2)
        gt_trajectories = batch['gt_trajectories'].to(device)  # (B, T, N, 2)
        
        optimizer.zero_grad()
        
        # Forward pass with dictionary output
        output = model(
            frames, initial_points, 
            return_rays=use_rays,
            return_attention=True
        )
        
        # Extract predictions from dictionary
        pred_points = output['points']
        pred_rays = output.get('rays', None)
        pred_visibility = output.get('visibility', None)  # (B, T, N)
        attention_weights = output.get('attention', None)
        
        # Get ground truth visibility
        gt_visibility = batch.get('gt_visibilities', None)
        if gt_visibility is not None:
            gt_visibility = gt_visibility.to(device)  # (B, T, N)
        
        # Compute ground truth rays from 3D trajectories
        gt_rays = None
        if use_rays and 'gt_trajectories_3d' in batch and 'gt_extrinsics' in batch:
            gt_trajs_3d = batch['gt_trajectories_3d'].to(device)  # (B, T, N, 3)
            gt_extrinsics = batch['gt_extrinsics'].to(device)  # (B, T, 4, 4)
            
            # Extract camera center from extrinsics
            gt_camera_params = extract_camera_parameters_from_extrinsics(gt_extrinsics)
            gt_camera_center = gt_camera_params['center']  # (B, T, 3)
            
            # Compute ground truth rays
            gt_rays = compute_ground_truth_rays(gt_trajs_3d, gt_camera_center)
        
        # Compute losses (ray loss implicitly supervises camera parameters)
        losses = compute_loss(
            pred_points, gt_trajectories, initial_points,
            pred_rays=pred_rays, 
            gt_rays=gt_rays,
            pred_visibility=pred_visibility,
            gt_visibility=gt_visibility,
            lambda_ray=lambda_ray,
            lambda_visibility=0.1  # Weight for visibility loss
        )
        
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        num_batches = accumulate_losses(total_losses, losses, num_batches)
        
        # Log batch losses for plotting
        for key, value in losses.items():
            if key not in batch_losses_history:
                batch_losses_history[key] = []
            batch_losses_history[key].append(value.item())
        
        # Progress bar
        postfix = {
            'loss': f"{losses['total_loss'].item():.4f}",
            'epe': f"{losses['epe'].item():.4f}"
        }
        if use_rays:
            postfix['ray'] = f"{losses['ray_loss'].item():.4f}"
            postfix['dir'] = f"{losses['direction_loss'].item():.4f}"
            postfix['mom'] = f"{losses['moment_loss'].item():.4f}"
        if pred_visibility is not None:
            postfix['vis'] = f"{losses['visibility_loss'].item():.4f}"
        pbar.set_postfix(postfix)
        
        # Visualize predictions every 50 batches
        if batch_idx % 50 == 0 and batch_idx > 0:
            if attention_weights is not None:
                visualize_attention(
                    attention_weights,
                    frames[0].detach().cpu().numpy(),
                    save_path=save_dir / f'attention_epoch{epoch}_batch{batch_idx}.png'
                )
            
            visualize_predictions(
                frames[0].detach().cpu().numpy(),
                pred_points[0].detach().cpu().numpy(),
                gt_trajectories[0].detach().cpu().numpy(),
                initial_points[0].detach().cpu().numpy(),
                save_path=save_dir / f'train_predictions_epoch{epoch}_batch{batch_idx}.png'
            )
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


def validate(model, dataloader, device, epoch, save_dir, use_rays=False, lambda_ray=0.1):
    """Validate the model"""
    model.eval()
    total_losses = {
        'total_loss': 0, 'l1_loss': 0, 'epe': 0, 'temporal_loss': 0,
        'ray_loss': 0, 'direction_loss': 0, 'moment_loss': 0,
        'visibility_loss': 0
    }
    num_batches = 0
    
    # Store predictions for visualization
    all_pred_points = []
    all_gt_points = []
    all_frames = []
    all_initial_points = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Validation {epoch}')):
            frames = batch['frames'].to(device)
            initial_points = batch['initial_points'].to(device)
            gt_trajectories = batch['gt_trajectories'].to(device)
            
            # Forward pass with dictionary output
            output = model(
                frames, initial_points,
                return_rays=use_rays,
                return_attention=True
            )
            
            # Extract predictions from dictionary
            pred_points = output['points']
            pred_rays = output.get('rays', None)
            pred_visibility = output.get('visibility', None)  # (B, T, N)
            attention_weights = output.get('attention', None)
            
            # Get ground truth visibility
            gt_visibility = batch.get('gt_visibilities', None)
            if gt_visibility is not None:
                gt_visibility = gt_visibility.to(device)  # (B, T, N)
            
            # Compute ground truth rays from 3D trajectories
            gt_rays = None
            if use_rays and 'gt_trajectories_3d' in batch and 'gt_extrinsics' in batch:
                gt_trajs_3d = batch['gt_trajectories_3d'].to(device)  # (B, T, N, 3)
                gt_extrinsics = batch['gt_extrinsics'].to(device)  # (B, T, 4, 4)
                
                # Extract camera center from extrinsics
                gt_camera_params = extract_camera_parameters_from_extrinsics(gt_extrinsics)
                gt_camera_center = gt_camera_params['center']  # (B, T, 3)
                
                # Compute ground truth rays
                gt_rays = compute_ground_truth_rays(gt_trajs_3d, gt_camera_center)
            
            # Compute losses
            losses = compute_loss(
                pred_points, gt_trajectories, initial_points,
                pred_rays=pred_rays,
                gt_rays=gt_rays,
                pred_visibility=pred_visibility,
                gt_visibility=gt_visibility,
                lambda_ray=lambda_ray,
                lambda_visibility=0.1  # Weight for visibility loss
            )
            
            num_batches = accumulate_losses(total_losses, losses, num_batches)
            
            # Store first few batches for visualization
            if batch_idx < 3:
                all_pred_points.append(pred_points[0].cpu())
                all_gt_points.append(gt_trajectories[0].cpu())
                all_frames.append(frames[0].cpu())
                all_initial_points.append(initial_points[0].cpu())
            
            # Visualize first batch attention
            if batch_idx == 0 and attention_weights is not None:
                visualize_attention(
                    attention_weights,
                    frames[0].detach().cpu().numpy(),
                    save_path=save_dir / f'val_attention_epoch{epoch}.png'
                )
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    # Visualize multiple validation predictions
    for i, (frames, pred, gt, initial) in enumerate(zip(all_frames, all_pred_points, all_gt_points, all_initial_points)):
        visualize_predictions(
            frames.numpy(),
            pred.numpy(),
            gt.numpy(),
            initial.numpy(),
            save_path=save_dir / f'val_predictions_epoch{epoch}_sample{i}.png'
        )
    
    return avg_losses


def main():
    parser = argparse.ArgumentParser(description='Train Point Tracker')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Point Odyssey dataset root')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sequence_length', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_frames_per_video', type=int, default=None,
                        help='Maximum number of frames to use per video (default: None = use all)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints, visualizations, and loss plots')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--video_dirs', type=str, default=None,
                        help='Comma-separated list of video directories to use (default: all)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture (default: resnet18)')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Disable pre-trained weights for backbone')
    parser.add_argument('--search_radius', type=int, default=4,
                        help='Search radius for cosine similarity matching (default: 4)')
    parser.add_argument('--use_rays', action='store_true',
                        help='Enable ray prediction (Plücker coordinates) - implicitly supervises camera parameters')
    parser.add_argument('--lambda_ray', type=float, default=0.1,
                        help='Weight for ray prediction loss (default: 0.1)')
    
    args = parser.parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = save_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = save_dir / 'loss_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print(f'Using Point Odyssey dataset from {args.data_root}')
    video_dirs = getattr(args, 'video_dirs', None)
    if video_dirs:
        video_dirs = video_dirs.split(',') if isinstance(video_dirs, str) else video_dirs
    
    full_dataset = PointOdysseyDataset(
        data_root=args.data_root,
        video_dirs=video_dirs,
        sequence_length=args.sequence_length,
        num_points=args.num_points,
        image_size=(args.image_size, args.image_size),
        max_frames_per_video=args.max_frames_per_video
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    model = PointTracker(
        feature_dim=256,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        num_points=args.num_points,
        dropout=0.1,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        search_radius=args.search_radius
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 0
    best_val_loss = float('inf')
    epoch_losses_history = {'train': [], 'val': []}
    batch_losses_history = {}  # Per-batch losses for detailed plotting
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epoch_losses_history = checkpoint.get('epoch_losses_history', {'train': [], 'val': []})
        batch_losses_history = checkpoint.get('batch_losses_history', {})
        print(f'Resumed from epoch {start_epoch}')
    
    print('Starting training...')
    if args.use_rays:
        print('Ray prediction enabled (implicitly supervises camera parameters)')
    print(f'Visualizations will be saved to: {vis_dir}')
    print(f'Loss plots will be saved to: {plots_dir}')
    
    for epoch in range(start_epoch, args.epochs):
        train_losses = train_epoch(
            model, train_loader, optimizer, device, epoch, vis_dir, batch_losses_history,
            use_rays=args.use_rays,
            lambda_ray=args.lambda_ray
        )
        epoch_losses_history['train'].append(train_losses)
        
        val_losses = validate(
            model, val_loader, device, epoch, vis_dir,
            use_rays=args.use_rays,
            lambda_ray=args.lambda_ray
        )
        epoch_losses_history['val'].append(val_losses)
        
        scheduler.step()
        
        print(f'\nEpoch {epoch}:')
        print(f'  Train Loss: {train_losses["total_loss"]:.4f}, EPE: {train_losses["epe"]:.4f}')
        if args.use_rays:
            print(f'  Train Ray Loss: {train_losses["ray_loss"]:.4f} '
                  f'(dir: {train_losses["direction_loss"]:.4f}, mom: {train_losses["moment_loss"]:.4f})')
        if 'visibility_loss' in train_losses and train_losses['visibility_loss'] > 0:
            print(f'  Train Visibility Loss: {train_losses["visibility_loss"]:.4f}')
        print(f'  Val Loss: {val_losses["total_loss"]:.4f}, EPE: {val_losses["epe"]:.4f}')
        if args.use_rays:
            print(f'  Val Ray Loss: {val_losses["ray_loss"]:.4f} '
                  f'(dir: {val_losses["direction_loss"]:.4f}, mom: {val_losses["moment_loss"]:.4f})')
        if 'visibility_loss' in val_losses and val_losses['visibility_loss'] > 0:
            print(f'  Val Visibility Loss: {val_losses["visibility_loss"]:.4f}')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'epoch_losses_history': epoch_losses_history,
            'batch_losses_history': batch_losses_history
        }
        
        torch.save(checkpoint, save_dir / 'latest.pth')
        
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, save_dir / 'best.pth')
            print(f'  Saved best model (val_loss: {best_val_loss:.4f})')
        
        # Plot detailed loss curves
        plot_detailed_losses(
            epoch_losses_history, 
            batch_losses_history, 
            save_path=plots_dir / f'losses_epoch{epoch}.png'
        )
    
    print('Training complete!')
    print(f'All visualizations saved to: {vis_dir}')
    print(f'Loss plots saved to: {plots_dir}')


if __name__ == '__main__':
    main()

