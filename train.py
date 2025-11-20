"""
Training script for Point Tracker with visualization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from model import PointTracker
from dataset import PointOdysseyDataset
from visualize import visualize_attention, plot_loss_history, visualize_predictions


def compute_loss(pred_points, gt_points, initial_points):
    """
    Compute tracking loss
    Args:
        pred_points: (B, T, N, 2) predicted points
        gt_points: (B, T, N, 2) ground truth points
        initial_points: (B, N, 2) initial points
    """
    # L2 loss on predicted trajectories
    l2_loss = nn.functional.mse_loss(pred_points, gt_points)
    
    # Endpoint error (EPE) - average distance error
    epe = torch.sqrt(torch.sum((pred_points - gt_points) ** 2, dim=-1))  # (B, T, N)
    epe_loss = epe.mean()
    
    # Temporal consistency loss (encourage smooth trajectories)
    if pred_points.shape[1] > 1:
        pred_velocities = pred_points[:, 1:] - pred_points[:, :-1]  # (B, T-1, N, 2)
        gt_velocities = gt_points[:, 1:] - gt_points[:, :-1]
        temporal_loss = nn.functional.mse_loss(pred_velocities, gt_velocities)
    else:
        temporal_loss = torch.tensor(0.0, device=pred_points.device)
    
    total_loss = l2_loss + 0.1 * temporal_loss
    
    return {
        'total_loss': total_loss,
        'l2_loss': l2_loss,
        'epe': epe_loss,
        'temporal_loss': temporal_loss
    }


def accumulate_losses(total_losses, losses, num_batches):
    """Helper function to accumulate losses"""
    for key in total_losses:
        total_losses[key] += losses[key].item()
    return num_batches + 1


def train_epoch(model, dataloader, optimizer, device, epoch, writer, save_dir):
    """Train for one epoch"""
    model.train()
    total_losses = {'total_loss': 0, 'l2_loss': 0, 'epe': 0, 'temporal_loss': 0}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        initial_points = batch['initial_points'].to(device)  # (B, N, 2)
        gt_trajectories = batch['gt_trajectories'].to(device)  # (B, T, N, 2)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get predictions with attention
        pred_points, attention_weights = model(
            frames, initial_points, return_attention=True
        )
        
        # Compute loss
        losses = compute_loss(pred_points, gt_trajectories, initial_points)
        
        # Backward pass
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        num_batches = accumulate_losses(total_losses, losses, num_batches)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'epe': f"{losses['epe'].item():.4f}"
        })
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        for key, value in losses.items():
            writer.add_scalar(f'Train/{key}', value.item(), global_step)
        
        # Visualize attention and predictions periodically
        if batch_idx % 50 == 0 and batch_idx > 0:
            # Visualize attention
            visualize_attention(
                attention_weights,
                frames[0].cpu().numpy(),
                save_path=save_dir / f'attention_epoch{epoch}_batch{batch_idx}.png'
            )
            
            # Visualize predictions
            visualize_predictions(
                frames[0].cpu().numpy(),
                pred_points[0].cpu().numpy(),
                gt_trajectories[0].cpu().numpy(),
                initial_points[0].cpu().numpy(),
                save_path=save_dir / f'predictions_epoch{epoch}_batch{batch_idx}.png'
            )
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


def validate(model, dataloader, device, epoch, writer, save_dir):
    """Validate the model"""
    model.eval()
    total_losses = {'total_loss': 0, 'l2_loss': 0, 'epe': 0, 'temporal_loss': 0}
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Validation {epoch}')):
            frames = batch['frames'].to(device)
            initial_points = batch['initial_points'].to(device)
            gt_trajectories = batch['gt_trajectories'].to(device)
            
            # Forward pass
            pred_points, attention_weights = model(
                frames, initial_points, return_attention=True
            )
            
            # Compute loss
            losses = compute_loss(pred_points, gt_trajectories, initial_points)
            
            # Accumulate losses
            num_batches = accumulate_losses(total_losses, losses, num_batches)
            
            # Visualize first batch
            if batch_idx == 0:
                visualize_attention(
                    attention_weights,
                    frames[0].cpu().numpy(),
                    save_path=save_dir / f'val_attention_epoch{epoch}.png'
                )
                visualize_predictions(
                    frames[0].cpu().numpy(),
                    pred_points[0].cpu().numpy(),
                    gt_trajectories[0].cpu().numpy(),
                    initial_points[0].cpu().numpy(),
                    save_path=save_dir / f'val_predictions_epoch{epoch}.png'
                )
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    # Log to tensorboard
    for key, value in avg_losses.items():
        writer.add_scalar(f'Val/{key}', value, epoch)
    
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
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--video_dirs', type=str, default=None,
                        help='Comma-separated list of video directories to use (default: all)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture (default: resnet18)')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Disable pre-trained weights for backbone')
    
    args = parser.parse_args()
    
    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = save_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset
    print(f'Using Point Odyssey dataset from {args.data_root}')
    # You can specify video_dirs or leave None to use all videos
    video_dirs = getattr(args, 'video_dirs', None)
    if video_dirs:
        video_dirs = video_dirs.split(',') if isinstance(video_dirs, str) else video_dirs
    
    full_dataset = PointOdysseyDataset(
        data_root=args.data_root,
        video_dirs=video_dirs,
        sequence_length=args.sequence_length,
        num_points=args.num_points,
        image_size=(args.image_size, args.image_size)
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
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
    
    # Create model
    model = PointTracker(
        feature_dim=256,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        num_points=args.num_points,
        dropout=0.1,
        backbone=args.backbone,
        pretrained=not args.no_pretrained
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    loss_history = {'train': [], 'val': []}
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        loss_history = checkpoint.get('loss_history', {'train': [], 'val': []})
        print(f'Resumed from epoch {start_epoch}')
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, device, epoch, writer, vis_dir
        )
        loss_history['train'].append(train_losses['total_loss'])
        
        # Validate
        val_losses = validate(model, val_loader, device, epoch, writer, vis_dir)
        loss_history['val'].append(val_losses['total_loss'])
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f'\nEpoch {epoch}:')
        print(f'  Train Loss: {train_losses["total_loss"]:.4f}, EPE: {train_losses["epe"]:.4f}')
        print(f'  Val Loss: {val_losses["total_loss"]:.4f}, EPE: {val_losses["epe"]:.4f}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'loss_history': loss_history
        }
        
        # Save latest
        torch.save(checkpoint, save_dir / 'latest.pth')
        
        # Save best
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, save_dir / 'best.pth')
            print(f'  Saved best model (val_loss: {best_val_loss:.4f})')
        
        # Plot loss history
        plot_loss_history(loss_history, save_path=vis_dir / f'loss_history_epoch{epoch}.png')
    
    print('Training complete!')
    writer.close()


if __name__ == '__main__':
    main()

