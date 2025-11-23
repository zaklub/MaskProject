#!/usr/bin/env python3
"""Test SAM-2 model loading to diagnose issues"""

import sys
from pathlib import Path
import torch

def test_sam2_loading():
    print("=" * 60)
    print("SAM-2 Loading Test")
    print("=" * 60)
    
    # Check imports
    print("\n[1] Testing imports...")
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✓ SAM-2 imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return 1
    
    # Check config file
    print("\n[2] Checking config file...")
    config_paths = [
        "sam2_hiera_l.yaml",
        Path(__file__).parent / "sam2_hiera_l.yaml",
    ]
    
    # Try to find config in SAM-2 installation
    try:
        import sam2
        sam2_path = Path(sam2.__file__).parent
        config_paths.extend([
            sam2_path / 'sam2_hiera_l.yaml',
            sam2_path / 'configs' / 'sam2' / 'sam2_hiera_l.yaml',
            sam2_path / 'configs' / 'sam2.1' / 'sam2.1_hiera_l.yaml',
        ])
    except:
        pass
    
    config_found = None
    for config_path in config_paths:
        path = Path(config_path)
        if path.exists():
            print(f"✓ Found config: {path.absolute()}")
            config_found = str(path.absolute())
            break
    
    if not config_found:
        print("✗ Config file not found in any location")
        print("  Searched:")
        for cp in config_paths:
            print(f"    - {cp}")
        return 1
    
    # Check model file
    print("\n[3] Checking model file...")
    model_paths = [
        "sam2_hiera_large.pt",
        Path(__file__).parent / "sam2_hiera_large.pt",
    ]
    
    model_found = None
    for model_path in model_paths:
        path = Path(model_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ Found model: {path.absolute()} ({size_mb:.1f} MB)")
            model_found = str(path.absolute())
            break
    
    if not model_found:
        print("✗ Model file not found")
        print("  Please download using: python download_sam2_models.py")
        return 1
    
    # Try loading model
    print("\n[4] Attempting to load SAM-2 model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        print(f"Config path: {config_found}")
        print(f"Model path: {model_found}")
        
        # Try with absolute paths
        model = build_sam2(config_found, model_found, device=device)
        print("✓ Model loaded successfully!")
        
        # Test predictor
        print("\n[5] Testing predictor...")
        predictor = SAM2ImagePredictor(model)
        print("✓ Predictor created successfully!")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! SAM-2 is working correctly.")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        print(f"\nError type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("Troubleshooting suggestions:")
        print("=" * 60)
        print("1. Verify SAM-2 installation:")
        print("   pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        print("\n2. Check config file format (should be YAML)")
        print("\n3. Verify model file is complete (should be ~1-2 GB)")
        print("\n4. Try reinstalling SAM-2:")
        print("   pip uninstall sam2")
        print("   pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        return 1

if __name__ == '__main__':
    sys.exit(test_sam2_loading())

