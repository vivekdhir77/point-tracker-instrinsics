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
