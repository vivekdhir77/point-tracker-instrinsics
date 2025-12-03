"""
Inference script for Point Tracker

Usage:
    # Random points (default - fastest way to test)
    python inference.py --checkpoint best.pth --frames_dir ./my_video/frames --random

    # Interactive mode - click points on first frame
    python inference.py --checkpoint best.pth --frames_dir ./my_video/frames

    # From file - provide point coordinates
    python inference.py --checkpoint best.pth --frames_dir ./my_video/frames --points points.txt

    # From video file
    python inference.py --checkpoint best.pth --video ./my_video.mp4 --random --num_points 16

    # With ray prediction and camera recovery
    python inference.py --checkpoint best.pth --frames_dir ./frames --random --predict_rays --predict_camera

    # Reproducible random points with seed
    python inference.py --checkpoint best.pth --frames_dir ./frames --random --seed 42
"""

import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from model import PointTracker


def load_frames_from_directory(frames_dir, image_size=(256, 256), max_frames=None):
    """Load frames from a directory"""
    frames_dir = Path(frames_dir)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    frame_files = []
    for ext in image_extensions:
        frame_files.extend(sorted(frames_dir.glob(ext)))
    
    if len(frame_files) == 0:
        raise ValueError(f"No image files found in {frames_dir}")
    
    print(f"Found {len(frame_files)} frames")
    
    if max_frames:
        frame_files = frame_files[:max_frames]
    
    frames = []
    original_sizes = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert('RGB')
        original_sizes.append(img.size)  # (W, H)
        
        # Resize
        img_resized = img.resize(image_size)
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # (C, H, W)
        frames.append(img_array)
    
    frames = np.stack(frames, axis=0)  # (T, C, H, W)
    
    return frames, original_sizes, [str(f) for f in frame_files]


def load_frames_from_video(video_path, image_size=(256, 256), max_frames=None):
    """Load frames from a video file"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    original_sizes = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_sizes.append((frame.shape[1], frame.shape[0]))  # (W, H)
        
        # Resize
        frame_resized = cv2.resize(frame, image_size)
        frame_array = frame_resized.astype(np.float32) / 255.0
        frame_array = frame_array.transpose(2, 0, 1)  # (C, H, W)
        frames.append(frame_array)
        
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    print(f"Extracted {len(frames)} frames from video")
    frames = np.stack(frames, axis=0)  # (T, C, H, W)
    
    return frames, original_sizes, [f"frame_{i:06d}" for i in range(len(frames))]


def load_query_points_from_file(points_file):
    """
    Load query points from file
    Format: one point per line as "x,y" (normalized [0, 1] or pixel coordinates)
    """
    points = []
    with open(points_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                x, y = map(float, line.split(','))
                points.append([x, y])
    
    return np.array(points, dtype=np.float32)


def generate_random_points(num_points=8, margin=0.1, seed=None):
    """
    Generate random query points
    
    Args:
        num_points: number of points to generate
        margin: margin from edges (in normalized coordinates)
        seed: random seed for reproducibility
    
    Returns:
        points: (N, 2) normalized coordinates [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate points with margin from edges
    points = np.random.uniform(margin, 1.0 - margin, size=(num_points, 2))
    
    return points.astype(np.float32)


def select_points_interactive(first_frame, num_points=8, image_size=(256, 256)):
    """
    Interactively select query points by clicking on the first frame
    
    Args:
        first_frame: (C, H, W) first frame
        num_points: number of points to select
        image_size: (H, W) frame size
    
    Returns:
        points: (N, 2) normalized coordinates [0, 1]
    """
    print(f"\n{'='*60}")
    print(f"INTERACTIVE POINT SELECTION")
    print(f"{'='*60}")
    print(f"Click {num_points} points on the image")
    print("Press 'r' to reset, 'q' to quit, Enter to confirm")
    print(f"{'='*60}\n")
    
    # Convert to display format
    frame_display = first_frame.transpose(1, 2, 0)  # (H, W, C)
    frame_display = (frame_display * 255).astype(np.uint8)
    
    points = []
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(frame_display)
    ax.set_title(f'Click {num_points} points (current: 0/{num_points})')
    
    scatter = ax.scatter([], [], c='red', s=100, marker='x', linewidths=3)
    
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            if len(points) < num_points:
                # Normalize coordinates to [0, 1]
                x_norm = event.xdata / image_size[1]  # width
                y_norm = event.ydata / image_size[0]  # height
                points.append([x_norm, y_norm])
                
                # Update display
                points_array = np.array(points)
                scatter.set_offsets(points_array * np.array([image_size[1], image_size[0]]))
                ax.set_title(f'Click {num_points} points (current: {len(points)}/{num_points})')
                fig.canvas.draw()
                
                print(f"Point {len(points)}: ({x_norm:.4f}, {y_norm:.4f})")
                
                if len(points) == num_points:
                    print(f"\nAll {num_points} points selected! Close the window or press Enter to continue.")
    
    def onkey(event):
        if event.key == 'r':
            points.clear()
            scatter.set_offsets(np.empty((0, 2)))
            ax.set_title(f'Click {num_points} points (current: 0/{num_points})')
            fig.canvas.draw()
            print("Reset points")
        elif event.key == 'enter':
            plt.close(fig)
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
    
    if len(points) == 0:
        raise ValueError("No points selected!")
    
    return np.array(points, dtype=np.float32)


def visualize_tracking_results(frames, predicted_points, query_points, save_path, 
                               frame_names=None, show_trails=True):
    """
    Visualize tracking results across all frames
    
    Args:
        frames: (T, C, H, W) frames
        predicted_points: (T, N, 2) predicted point trajectories (normalized)
        query_points: (N, 2) initial query points
        save_path: path to save visualization
        frame_names: list of frame names
        show_trails: whether to show point trails
    """
    T, C, H, W = frames.shape
    N = predicted_points.shape[1]
    
    # Create color map for points
    colors = plt.cm.rainbow(np.linspace(0, 1, N))
    
    # Determine grid size
    cols = min(4, T)
    rows = (T + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    fig.suptitle('Point Tracking Results', fontsize=16, fontweight='bold')
    
    for t in range(T):
        row = t // cols
        col = t % cols
        ax = axes[row, col]
        
        # Display frame
        frame_display = frames[t].transpose(1, 2, 0)  # (H, W, C)
        frame_display = np.clip(frame_display, 0, 1)
        ax.imshow(frame_display)
        
        # Plot points
        points_t = predicted_points[t]  # (N, 2)
        points_pixel = points_t * np.array([W, H])
        
        for n in range(N):
            # Plot current point
            ax.plot(points_pixel[n, 0], points_pixel[n, 1], 
                   'o', color=colors[n], markersize=10, 
                   markeredgecolor='white', markeredgewidth=2)
            
            # Plot trail from beginning to current frame
            if show_trails and t > 0:
                trail = predicted_points[:t+1, n] * np.array([W, H])
                ax.plot(trail[:, 0], trail[:, 1], 
                       '-', color=colors[n], linewidth=2, alpha=0.6)
            
            # Label point number on first frame
            if t == 0:
                ax.text(points_pixel[n, 0] + 5, points_pixel[n, 1] - 5,
                       str(n+1), color='white', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[n], alpha=0.7))
        
        title = f'Frame {t}'
        if frame_names:
            title += f'\n{Path(frame_names[t]).name}'
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for t in range(T, rows * cols):
        row = t // cols
        col = t % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved tracking visualization: {save_path}")


def save_results_json(predicted_points, query_points, frame_names, save_path, 
                     camera_params=None, rays=None):
    """Save tracking results to JSON file"""
    results = {
        'num_frames': len(frame_names),
        'num_points': len(query_points),
        'query_points': query_points.tolist(),
        'frame_names': frame_names,
        'trajectories': predicted_points.tolist(),  # (T, N, 2)
    }
    
    if camera_params is not None:
        results['camera_parameters'] = {
            'center': camera_params['center'].tolist(),
            'rotation': camera_params['rotation'].tolist(),
            'intrinsics': camera_params['intrinsics'].tolist(),
            'translation': camera_params['translation'].tolist()
        }
    
    if rays is not None:
        results['rays'] = rays.tolist()  # (T, N, 6)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to JSON: {save_path}")


def save_results_txt(predicted_points, query_points, frame_names, save_path):
    """Save tracking results to simple text file"""
    with open(save_path, 'w') as f:
        f.write(f"# Point Tracking Results\n")
        f.write(f"# Num frames: {len(frame_names)}\n")
        f.write(f"# Num points: {len(query_points)}\n")
        f.write(f"#\n")
        f.write(f"# Format: frame_idx, point_idx, x, y\n")
        f.write(f"#\n")
        
        for t, frame_name in enumerate(frame_names):
            for n in range(len(query_points)):
                x, y = predicted_points[t, n]
                f.write(f"{t}, {n}, {x:.6f}, {y:.6f}\n")
    
    print(f"Saved results to TXT: {save_path}")


def run_inference(args):
    """Main inference function"""
    print("\n" + "="*60)
    print("POINT TRACKER INFERENCE")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Initialize model
    model = PointTracker(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_points=args.num_points,
        dropout=0.0,  # No dropout for inference
        backbone=args.backbone,
        pretrained=False,
        use_correlation_matching=args.use_correlation_matching,
        search_radius=args.search_radius
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully\n")
    
    # Load frames
    print("Loading frames...")
    if args.video:
        frames, original_sizes, frame_names = load_frames_from_video(
            args.video, image_size=(args.image_size, args.image_size), 
            max_frames=args.max_frames
        )
    else:
        frames, original_sizes, frame_names = load_frames_from_directory(
            args.frames_dir, image_size=(args.image_size, args.image_size),
            max_frames=args.max_frames
        )
    
    T = len(frames)
    print(f"Loaded {T} frames of size {args.image_size}x{args.image_size}\n")
    
    # Get query points
    if args.points:
        print(f"Loading query points from: {args.points}")
        query_points = load_query_points_from_file(args.points)
        
        # Check if points are in pixel coordinates (> 1) and normalize
        if query_points.max() > 1.0:
            print(f"Points appear to be in pixel coordinates, normalizing...")
            W, H = original_sizes[0]
            query_points[:, 0] /= W
            query_points[:, 1] /= H
    elif args.random:
        print(f"Generating {args.num_points} random query points...")
        query_points = generate_random_points(
            num_points=args.num_points,
            margin=args.margin,
            seed=args.seed
        )
    else:
        print("Launching interactive point selection...")
        query_points = select_points_interactive(
            frames[0], 
            num_points=args.num_points,
            image_size=(args.image_size, args.image_size)
        )
    
    N = len(query_points)
    print(f"\nQuery points ({N} points):")
    for i, (x, y) in enumerate(query_points):
        print(f"  Point {i+1}: ({x:.4f}, {y:.4f})")
    
    # Prepare input
    frames_tensor = torch.from_numpy(frames).unsqueeze(0).float().to(device)  # (1, T, C, H, W)
    query_points_tensor = torch.from_numpy(query_points).unsqueeze(0).float().to(device)  # (1, N, 2)
    
    # Run inference
    print(f"\nRunning inference on {T} frames...")
    with torch.no_grad():
        output = model(
            frames_tensor,
            query_points_tensor,
            return_rays=args.predict_rays,
            return_camera=args.predict_camera,
            return_attention=False
        )
    
    # Extract results
    predicted_points = output['points'].squeeze(0).cpu().numpy()  # (T, N, 2)
    print(f"✓ Predicted trajectories: {predicted_points.shape}")
    
    rays = None
    if args.predict_rays and 'rays' in output:
        rays = output['rays'].squeeze(0).cpu().numpy()  # (T, N, 6)
        print(f"✓ Predicted rays: {rays.shape}")
    
    camera_params = None
    if args.predict_camera and 'camera' in output:
        camera_params = {
            'center': output['camera']['center'].squeeze(0).cpu().numpy(),  # (T, 3)
            'rotation': output['camera']['rotation'].squeeze(0).cpu().numpy(),  # (T, 3, 3)
            'intrinsics': output['camera']['intrinsics'].squeeze(0).cpu().numpy(),  # (T, 3, 3)
            'translation': output['camera']['translation'].squeeze(0).cpu().numpy()  # (T, 3)
        }
        print(f"✓ Recovered camera parameters")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize results
    print(f"\nGenerating visualizations...")
    vis_path = output_dir / 'tracking_visualization.png'
    visualize_tracking_results(
        frames, predicted_points, query_points, vis_path,
        frame_names=frame_names, show_trails=args.show_trails
    )
    
    # Save results
    print(f"\nSaving results...")
    save_results_json(
        predicted_points, query_points, frame_names,
        output_dir / 'results.json',
        camera_params=camera_params, rays=rays
    )
    
    save_results_txt(
        predicted_points, query_points, frame_names,
        output_dir / 'results.txt'
    )
    
    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"  - tracking_visualization.png")
    print(f"  - results.json")
    print(f"  - results.txt")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Point Tracker Inference')
    
    # Input/Output
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--frames_dir', type=str, default=None,
                        help='Directory containing image frames')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file (alternative to frames_dir)')
    parser.add_argument('--points', type=str, default=None,
                        help='Path to query points file (if not provided, uses random or interactive)')
    parser.add_argument('--random', action='store_true',
                        help='Use random query points instead of interactive selection')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Margin from edges for random points (default: 0.1)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible random points')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save results')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_points', type=int, default=8,
                        help='Number of points (used for interactive selection)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--use_correlation_matching', action='store_true', default=True)
    parser.add_argument('--search_radius', type=int, default=4)
    
    # Processing options
    parser.add_argument('--image_size', type=int, default=256,
                        help='Resize frames to this size')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process')
    parser.add_argument('--predict_rays', action='store_true',
                        help='Predict rays (Plücker coordinates)')
    parser.add_argument('--predict_camera', action='store_true',
                        help='Recover camera parameters')
    parser.add_argument('--show_trails', action='store_true', default=True,
                        help='Show point trails in visualization')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.frames_dir and not args.video:
        parser.error("Must provide either --frames_dir or --video")
    
    if args.frames_dir and args.video:
        parser.error("Provide only one of --frames_dir or --video")
    
    run_inference(args)


if __name__ == '__main__':
    main()

