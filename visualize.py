"""
Visualization utilities for attention and training progress
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch


def visualize_attention(attention_weights_list, frames, save_path=None, layer_idx=0, head_idx=0):
    """
    Visualize attention weights from transformer layers
    Args:
        attention_weights_list: List of attention weights from each layer
            Each element is (B, num_heads, N, N) where N is the sequence length
            With frame-wise temporal tokens, N = T_prev + T*(N+2) or similar
        frames: (T, C, H, W) video frames
        save_path: Path to save visualization
        layer_idx: Which layer to visualize
        head_idx: Which attention head to visualize
    """
    if len(attention_weights_list) == 0:
        return
    
    # Get attention from specified layer and head
    if layer_idx >= len(attention_weights_list):
        layer_idx = len(attention_weights_list) - 1
    
    attn = attention_weights_list[layer_idx][0]  # (num_heads, N, N)
    
    if head_idx >= attn.shape[0]:
        head_idx = 0
    
    attn = attn[head_idx].detach().cpu().numpy()  # (N, N)
    
    T, C, H, W = frames.shape
    seq_len = attn.shape[0]
    
    # With frame-wise temporal tokens, the sequence structure is:
    # [prev_tokens..., temp_0, frame0, pts0, temp_1, frame1, pts1, ...]
    # For training, T_prev=0, so seq_len = T*(N+2) where N is num_points
    # We can't easily extract point-to-point attention without knowing the exact structure
    
    # Just visualize the full attention matrix
    temporal_attn = attn  # Use full attention matrix
    
    # Create a simpler visualization: frames on top, attention matrix below
    num_cols = max(T, 2)
    fig, axes = plt.subplots(2, num_cols, figsize=(3*num_cols, 6))
    if T == 1:
        axes = axes.reshape(2, 1)
    elif T == 0:
        axes = axes.reshape(2, 2)
    else:
        axes = axes.reshape(2, num_cols)
    
    # Show frames
    for t in range(T):
        if t < num_cols:
            frame = frames[t].transpose(1, 2, 0)  # (H, W, C)
            frame = np.clip(frame, 0, 1)
            axes[0, t].imshow(frame)
            axes[0, t].set_title(f'Frame {t}')
            axes[0, t].axis('off')
    
    # Hide unused frame axes
    for t in range(T, num_cols):
        axes[0, t].axis('off')
    
    # Show full attention matrix
    im = axes[1, 0].imshow(temporal_attn, cmap='hot', aspect='auto', interpolation='nearest')
    axes[1, 0].set_title(f'Attention Matrix\n(seq_len={seq_len})')
    axes[1, 0].set_xlabel('Key Position')
    axes[1, 0].set_ylabel('Query Position')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Hide unused attention axes
    for t in range(1, num_cols):
        axes[1, t].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_predictions(frames, pred_points, gt_points, initial_points, save_path=None):
    """
    Visualize predicted point trajectories
    Args:
        frames: (T, C, H, W) video frames
        pred_points: (T, N, 2) predicted points (normalized)
        gt_points: (T, N, 2) ground truth points (normalized)
        initial_points: (N, 2) initial points (normalized)
        save_path: Path to save visualization
    """
    T, C, H, W = frames.shape
    N = pred_points.shape[1]
    
    pred_pixel = pred_points.copy()
    pred_pixel[:, :, 0] *= W
    pred_pixel[:, :, 1] *= H
    
    gt_pixel = gt_points.copy()
    gt_pixel[:, :, 0] *= W
    gt_pixel[:, :, 1] *= H
    
    initial_pixel = initial_points.copy()
    initial_pixel[:, 0] *= W
    initial_pixel[:, 1] *= H
    
    fig, axes = plt.subplots(2, (T + 1) // 2, figsize=(4 * ((T + 1) // 2), 8))
    if T == 1:
        axes = axes.reshape(2, 1)
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, N))
    
    for t in range(T):
        ax = axes[t]
        frame = frames[t].transpose(1, 2, 0)  # (H, W, C)
        frame = np.clip(frame, 0, 1)
        ax.imshow(frame)
        
        for n in range(N):
            color = colors[n]
            
            if t > 0:
                gt_traj = gt_pixel[:t+1, n, :]
                ax.plot(gt_traj[:, 0], gt_traj[:, 1], '--', color=color, alpha=0.5, linewidth=1, label=f'GT {n}' if t == 0 else '')
            
            if t > 0:
                pred_traj = pred_pixel[:t+1, n, :]
                ax.plot(pred_traj[:, 0], pred_traj[:, 1], '-', color=color, linewidth=2, label=f'Pred {n}' if t == 0 else '')
            
            ax.scatter(gt_pixel[t, n, 0], gt_pixel[t, n, 1], 
                      c=[color], marker='o', s=50, edgecolors='white', linewidths=1, label=f'GT {n}' if t == 0 and n == 0 else '')
            ax.scatter(pred_pixel[t, n, 0], pred_pixel[t, n, 1], 
                      c=[color], marker='x', s=100, linewidths=2, label=f'Pred {n}' if t == 0 and n == 0 else '')
        
        if t == 0:
            for n in range(N):
                ax.scatter(initial_pixel[n, 0], initial_pixel[n, 1], 
                          c=[colors[n]], marker='*', s=200, edgecolors='black', linewidths=2, label=f'Init {n}' if n == 0 else '')
        
        ax.set_title(f'Frame {t}')
        ax.axis('off')
    
    for t in range(T, len(axes)):
        axes[t].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_history(loss_history, save_path=None):
    """
    Plot training and validation loss history
    Args:
        loss_history: Dict with 'train' and 'val' keys containing loss lists
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(loss_history['train']) + 1)
    
    axes.plot(epochs, loss_history['train'], 'b-', label='Train Loss', linewidth=2)
    axes.plot(epochs, loss_history['val'], 'r-', label='Val Loss', linewidth=2)
    
    axes.set_xlabel('Epoch', fontsize=12)
    axes.set_ylabel('Loss', fontsize=12)
    axes.set_title('Training and Validation Loss', fontsize=14)
    axes.legend(fontsize=11)
    axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_attention_heads(attention_weights_list, save_path=None, layer_idx=0):
    """
    Visualize all attention heads from a layer
    Args:
        attention_weights_list: List of attention weights
        save_path: Path to save visualization
        layer_idx: Which layer to visualize
    """
    if len(attention_weights_list) == 0:
        return
    
    attn = attention_weights_list[layer_idx][0]  # (num_heads, N, N)
    num_heads = attn.shape[0]
    
    cols = 4
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if num_heads > 1 else [axes]
    
    for head_idx in range(num_heads):
        attn_head = attn[head_idx].detach().cpu().numpy()
        im = axes[head_idx].imshow(attn_head, cmap='hot', aspect='auto')
        axes[head_idx].set_title(f'Head {head_idx}')
        plt.colorbar(im, ax=axes[head_idx])
    
    for head_idx in range(num_heads, len(axes)):
        axes[head_idx].axis('off')
    
    plt.suptitle(f'Attention Heads - Layer {layer_idx}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

