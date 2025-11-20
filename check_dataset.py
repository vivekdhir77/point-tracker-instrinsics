"""
Script to check if Point Odyssey dataset is properly formatted
"""
import argparse
from pathlib import Path
import numpy as np


def check_dataset(data_root):
    """Check if dataset structure is correct"""
    data_root = Path(data_root)
    
    if not data_root.exists():
        print(f"❌ Error: Data root directory does not exist: {data_root}")
        return False
    
    print(f"✓ Data root exists: {data_root}")
    
    # Find all video directories
    video_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    
    if len(video_dirs) == 0:
        print(f"❌ Error: No video directories found in {data_root}")
        return False
    
    print(f"✓ Found {len(video_dirs)} video directories")
    
    valid_videos = 0
    total_sequences = 0
    
    for video_dir in video_dirs:
        frames_dir = video_dir / "frames"
        anno_path = video_dir / "anno.npz"
        
        # Check frames directory
        if not frames_dir.exists():
            print(f"⚠️  Warning: Missing frames directory: {frames_dir}")
            continue
        
        # Check annotation file
        if not anno_path.exists():
            print(f"⚠️  Warning: Missing annotation file: {anno_path}")
            continue
        
        # Check frames
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        if len(frame_files) == 0:
            frame_files = sorted(frames_dir.glob("frame_*.png"))
        
        if len(frame_files) == 0:
            print(f"⚠️  Warning: No frame files found in {frames_dir}")
            continue
        
        # Check annotation
        try:
            anno = np.load(anno_path)
            
            if 'trajs_2d' not in anno:
                print(f"⚠️  Warning: Missing 'trajs_2d' in {anno_path}")
                continue
            
            if 'visibs' not in anno:
                print(f"⚠️  Warning: Missing 'visibs' in {anno_path}")
                continue
            
            trajs_2d = anno['trajs_2d']
            visibs = anno['visibs']
            
            print(f"\n✓ Video: {video_dir.name}")
            print(f"  - Frames: {len(frame_files)}")
            print(f"  - Annotation frames: {trajs_2d.shape[0]}")
            print(f"  - Total points: {trajs_2d.shape[1]}")
            print(f"  - Trajectory shape: {trajs_2d.shape}")
            print(f"  - Visibility shape: {visibs.shape}")
            
            # Check valid points in first frame
            first_frame_visibs = visibs[0]
            valid_points = np.sum(first_frame_visibs)
            print(f"  - Valid points in first frame: {valid_points}")
            
            # Calculate possible sequences (assuming sequence_length=8)
            min_frames = min(len(frame_files), trajs_2d.shape[0])
            sequence_length = 8
            if min_frames >= sequence_length:
                num_sequences = min_frames - sequence_length + 1
                total_sequences += num_sequences
                print(f"  - Possible sequences (length={sequence_length}): {num_sequences}")
            
            valid_videos += 1
            
        except Exception as e:
            print(f"⚠️  Warning: Error loading {anno_path}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  - Valid videos: {valid_videos}/{len(video_dirs)}")
    print(f"  - Total sequences (estimated): {total_sequences}")
    
    if valid_videos > 0:
        print(f"\n✓ Dataset structure looks good! You can start training.")
        print(f"\nTo train, run:")
        print(f"  python train.py --data_root {data_root}")
        return True
    else:
        print(f"\n❌ No valid videos found. Please check your dataset structure.")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check Point Odyssey dataset structure')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Point Odyssey dataset root')
    
    args = parser.parse_args()
    check_dataset(args.data_root)

