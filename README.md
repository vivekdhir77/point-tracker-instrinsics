## Model Features

### Point Tracker with Depth Prediction

The model now supports:
- **Point Tracking**: Track multiple points across video frames
- **Depth Prediction**: Predict depth values for each tracked point (optional)
- **Learnable Temporal Tokens**: Handle sequences up to 1000 frames
- **Attention Mechanism**: Multi-head self-attention for temporal reasoning

### Usage Example

```python
from model import PointTracker
import torch

# Initialize model
model = PointTracker(
    feature_dim=256,
    hidden_dim=256,
    num_heads=8,
    num_layers=4,
    num_points=8,
    max_temporal_len=1000,
    use_learnable_temporal=True
)

# Input: video frames and initial points
frames = torch.randn(1, 8, 3, 256, 256)  # (B, T, C, H, W)
initial_points = torch.rand(1, 8, 2)  # (B, N, 2) - normalized [0, 1]

# Forward pass - track points only
predicted_points = model(frames, initial_points)  # (B, T, N, 2)

# Forward pass - with depth prediction
predicted_points, predicted_depth = model(
    frames, initial_points, return_depth=True
)
# predicted_points: (B, T, N, 2)
# predicted_depth: (B, T, N, 1)

# Forward pass - with attention weights
predicted_points, attention_weights = model(
    frames, initial_points, return_attention=True
)

# Forward pass - with both depth and attention
predicted_points, predicted_depth, attention_weights = model(
    frames, initial_points, return_depth=True, return_attention=True
)
```

## Training

### Basic Training 

```bash
python train.py --data_root /path/to/point_odyssey_dataset
```

### Training with Custom Parameters

```bash
python train.py \
    --data_root /mnt/data/vivek/point_odyssey_v1.2/train/ \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --sequence_length 8 \
    --num_points 8 \
    --image_size 256 \
    --num_workers 4 \
    --save_dir ./checkpoints \
    --log_dir ./logs \
    --backbone resnet18
```

### Training with Different Backbone

Use a larger backbone for potentially better performance:

```bash
python train.py \
    --data_root /mnt/data/vivek/point_odyssey_v1.2/train/ \
    --backbone resnet50
```

Or train from scratch without pre-trained weights:

```bash
python train.py \
    --data_root /path/to/point_odyssey_dataset \
    --backbone resnet18 \
    --no_pretrained
```

### Training with Learnable Temporal Tokens for Long Context

The model now supports learnable temporal embeddings (enabled by default) that can handle longer temporal sequences up to 1000 frames:

```bash
python train.py \
    --data_root /path/to/point_odyssey_dataset \
    --sequence_length 16 \
    --max_temporal_len 1000
```

To use traditional sinusoidal temporal encodings instead:

```bash
python train.py \
    --data_root /path/to/point_odyssey_dataset \
    --no_learnable_temporal
```

### Training with Depth Prediction

The model includes a depth prediction head for multi-task learning. Enable it to predict depth values for each tracked point:

```bash
python train.py \
    --data_root /path/to/point_odyssey_dataset \
    --use_depth
```

Note: This requires your dataset to include depth ground truth data (`gt_depth` field in the dataset). The depth head predicts a depth value for each tracked point at each frame (B, T, N, 1).

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

## Training Arguments

- `--data_root`: Path to Point Odyssey dataset root directory (required)
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
- `--backbone`: Backbone architecture - `resnet18`, `resnet34`, or `resnet50` (default: resnet18)
- `--no_pretrained`: Disable pre-trained weights for backbone (train from scratch)
- `--max_temporal_len`: Maximum temporal length to support for learnable temporal tokens (default: 1000)
- `--no_learnable_temporal`: Disable learnable temporal tokens and use sinusoidal encoding instead
- `--use_depth`: Enable depth prediction head for multi-task learning (requires depth ground truth in dataset)
