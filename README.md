# Point Tracker

A transformer-based point tracking model for tracking points across video sequences, trained on the Point Odyssey dataset.

## Features

- Transformer-based architecture with multi-head attention
- Supports Point Odyssey dataset format
- Synthetic dataset for testing
- Visualization tools for attention and predictions
- TensorBoard logging
- Checkpoint saving and resuming

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The Point Odyssey dataset should be organized as follows:

```
data_root/
    video_dir_1/
        frames/
            frame_000000.jpg
            frame_000001.jpg
            ...
        anno.npz  # Contains 'trajs_2d' (T, N, 2) and 'visibs' (T, N)
    video_dir_2/
        frames/
            frame_000000.jpg
            ...
        anno.npz
    ...
```

The `anno.npz` file should contain:
- `trajs_2d`: numpy array of shape (T, N, 2) with 2D point trajectories in pixel coordinates
- `visibs`: numpy array of shape (T, N) with boolean visibility flags

## Training

### Basic Training on Point Odyssey Dataset

```bash
python train.py --data_root /path/to/point_odyssey_dataset
```

### Training with Custom Parameters

```bash
python train.py \
    --data_root /path/to/point_odyssey_dataset \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --sequence_length 8 \
    --num_points 8 \
    --image_size 256 \
    --num_workers 4 \
    --save_dir ./checkpoints \
    --log_dir ./logs
```

### Training on Specific Videos

To train on specific video directories only:

```bash
python train.py \
    --data_root /path/to/point_odyssey_dataset \
    --video_dirs video1,video2,video3
```

### Resume Training from Checkpoint

```bash
python train.py \
    --data_root /path/to/point_odyssey_dataset \
    --resume ./checkpoints/latest.pth
```

### Training with Synthetic Data (for testing)

```bash
python train.py --use_synthetic
```

## Training Arguments

- `--data_root`: Path to Point Odyssey dataset root directory
- `--use_synthetic`: Use synthetic dataset instead of Point Odyssey (for testing)
- `--batch_size`: Batch size (default: 4)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--sequence_length`: Number of frames per sequence (default: 8)
- `--num_points`: Number of points to track (default: 8)
- `--image_size`: Image size for resizing (default: 256)
- `--num_workers`: Number of data loading workers (default: 4)
- `--save_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--log_dir`: Directory for TensorBoard logs (default: ./logs)
- `--resume`: Path to checkpoint to resume from
- `--video_dirs`: Comma-separated list of video directories to use (default: all)

## Model Architecture

The model consists of:
- **Feature Extractor**: CNN backbone to extract features from video frames
- **Point Feature Extraction**: Bilinear interpolation to extract features at point locations
- **Transformer Layers**: Multi-head self-attention to model temporal and spatial relationships
- **Output Head**: Predicts point offsets for tracking

## Outputs

During training, the following are saved:

- **Checkpoints**: 
  - `checkpoints/latest.pth`: Latest model checkpoint
  - `checkpoints/best.pth`: Best model based on validation loss
  
- **Visualizations** (in `checkpoints/visualizations/`):
  - Attention visualizations
  - Prediction vs ground truth comparisons
  - Loss history plots

- **TensorBoard Logs**: Training and validation metrics in `logs/`

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./logs
```

## Testing

Test the model with synthetic data:

```bash
python test_model.py
```

## Model Parameters

Default model configuration:
- Feature dimension: 256
- Hidden dimension: 256
- Number of attention heads: 8
- Number of transformer layers: 4
- Dropout: 0.1

## Notes

- The dataset automatically splits into 80% train and 20% validation
- Points are normalized to [0, 1] range based on original image size
- Only points visible in the first frame are used for tracking
- The model uses gradient clipping (max_norm=1.0) for training stability
- Learning rate follows a cosine annealing schedule

