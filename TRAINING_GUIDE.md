# Quick Training Guide

## Train on 2 Specific Folders with Max 300 Frames Each

### Step 1: Identify Your Folders

List available folders in your dataset:
```bash
ls /mnt/data/vivek/point_odyssey_v1.2/train/
```

Let's say you want to use `ani2` and `ani3` (replace with your actual folder names).

### Step 2: Run Training

```bash
python train.py \
    --data_root /mnt/data/vivek/point_odyssey_v1.2/train/ \
    --video_dirs ani2,ani3 \
    --max_frames_per_video 300 \
    --batch_size 4 \
    --sequence_length 8 \
    --num_points 8 \
    --epochs 50 \
    --save_dir ./checkpoints_2videos
```

### What This Does

- **`--video_dirs ani2,ani3`**: Train on only these 2 folders
- **`--max_frames_per_video 300`**: Use maximum 300 frames from each folder
- **`--batch_size 4`**: Process 4 sequences at a time
- **`--sequence_length 8`**: Each sequence has 8 consecutive frames
- **`--num_points 8`**: Track 8 points per sequence

### With Ray Supervision

```bash
python train.py \
    --data_root /mnt/data/vivek/point_odyssey_v1.2/train/ \
    --video_dirs ani2,ani3 \
    --max_frames_per_video 300 \
    --use_rays \
    --lambda_ray 0.1 \
    --batch_size 4 \
    --epochs 100 \
    --save_dir ./checkpoints_rays_2videos
```

### Expected Output

During data loading:
```
Info: Limiting ani2 to 300 frames (max_frames_per_video=300)
Info: Limiting ani3 to 300 frames (max_frames_per_video=300)
```

Number of training sequences:
- Each video: max 300 frames
- Sequences per video: `(300 - sequence_length + 1)` = 293 possible starting positions
- With 2 videos: ~586 total sequences
- Split 80/20: ~469 train, ~117 validation

### Monitoring Training

Watch the loss plots in real-time:
```bash
# In another terminal
watch -n 5 "ls -lth checkpoints_2videos/loss_plots/"
```

View latest loss plot:
```bash
open checkpoints_2videos/loss_plots/losses_epoch0.png
```

### Output Structure

```
checkpoints_2videos/
├── best.pth                    # Best model checkpoint
├── latest.pth                  # Latest checkpoint
├── visualizations/             # Prediction visualizations
│   ├── train_predictions_epoch0_batch50.png
│   ├── val_predictions_epoch0_sample0.png
│   └── ...
└── loss_plots/                 # Loss curves
    ├── losses_epoch0.png       # 6-panel loss visualization
    ├── losses_epoch1.png
    └── ...
```

### Resume Training

If training is interrupted:
```bash
python train.py \
    --data_root /mnt/data/vivek/point_odyssey_v1.2/train/ \
    --video_dirs ani2,ani3 \
    --max_frames_per_video 300 \
    --resume ./checkpoints_2videos/latest.pth \
    --epochs 100
```

### After Training - Run Inference

```bash
python inference.py \
    --checkpoint checkpoints_2videos/best.pth \
    --frames_dir /path/to/test/video/frames \
    --random \
    --num_points 16 \
    --predict_rays \
    --output_dir ./inference_results
```

## Common Issues & Solutions

### Issue: "No sequences found"
**Cause**: Not enough visible points in first frame  
**Solution**: Dataset automatically skips frames without visible points - this is normal

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size:
```bash
--batch_size 2  # or even 1
```

### Issue: "Too slow"
**Solution**: 
1. Reduce image size: `--image_size 128`
2. Reduce sequence length: `--sequence_length 4`
3. Use fewer workers: `--num_workers 2`

### Issue: Ray loss is NaN
**Solution**:
1. Check if 3D data is loading: add print statements
2. Reduce `--lambda_ray`: try 0.01 or 0.001
3. Train without rays first: remove `--use_rays`

## Quick Reference Commands

### Minimal Training (2 folders, 300 frames each)
```bash
python train.py \
    --data_root /mnt/data/vivek/point_odyssey_v1.2/train/ \
    --video_dirs ani2,ani3 \
    --max_frames_per_video 300
```

### Full Training (with all options)
```bash
python train.py \
    --data_root /mnt/data/vivek/point_odyssey_v1.2/train/ \
    --video_dirs ani2,ani3 \
    --max_frames_per_video 300 \
    --use_rays \
    --lambda_ray 0.1 \
    --batch_size 8 \
    --sequence_length 8 \
    --num_points 16 \
    --image_size 256 \
    --epochs 100 \
    --backbone resnet50 \
    --lr 1e-4 \
    --save_dir ./checkpoints_final
```

### Fast Prototyping (small config)
```bash
python train.py \
    --data_root /mnt/data/vivek/point_odyssey_v1.2/train/ \
    --video_dirs ani2 \
    --max_frames_per_video 100 \
    --batch_size 2 \
    --sequence_length 4 \
    --image_size 128 \
    --epochs 10 \
    --save_dir ./quick_test
```

