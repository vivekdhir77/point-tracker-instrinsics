"""
Quick test script to verify the model works
"""
import torch
from model import PointTracker
from dataset import SyntheticDataset
from torch.utils.data import DataLoader

def test_model():
    """Test the model with synthetic data"""
    print("Testing Point Tracker Model...")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = PointTracker(
        feature_dim=256,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        num_points=8,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create synthetic dataset
    dataset = SyntheticDataset(
        num_samples=10,
        sequence_length=8,
        num_points=8,
        image_size=(256, 256)
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            frames = batch['frames'].to(device)  # (B, T, C, H, W)
            initial_points = batch['initial_points'].to(device)  # (B, N, 2)
            
            print(f"Input frames shape: {frames.shape}")
            print(f"Initial points shape: {initial_points.shape}")
            
            # Forward pass
            pred_points, attention_weights = model(
                frames, initial_points, return_attention=True
            )
            
            print(f"Predicted points shape: {pred_points.shape}")
            print(f"Number of attention layers: {len(attention_weights)}")
            print(f"Attention shape (first layer): {attention_weights[0].shape}")
            
            print("\n✓ Model test passed!")
            break

if __name__ == '__main__':
    test_model()

