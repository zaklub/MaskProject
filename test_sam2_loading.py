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
    
    # List all available config files in SAM-2 installation
    print("\n[3.5] Checking for other config files in SAM-2 installation...")
    try:
        import sam2
        sam2_path = Path(sam2.__file__).parent
        print(f"SAM-2 installation path: {sam2_path}")
        
        # Look for all YAML files
        config_dirs = [
            sam2_path,
            sam2_path / 'configs',
            sam2_path / 'configs' / 'sam2',
            sam2_path / 'configs' / 'sam2.1',
        ]
        
        yaml_files = []
        for config_dir in config_dirs:
            if config_dir.exists():
                for yaml_file in config_dir.glob("*.yaml"):
                    yaml_files.append(yaml_file)
        
        if yaml_files:
            print("Found YAML config files:")
            for yf in yaml_files[:10]:  # Show first 10
                print(f"  - {yf}")
        else:
            print("  No YAML files found in expected locations")
    except Exception as e:
        print(f"  Could not check SAM-2 installation: {e}")
    
    # Try loading model
    print("\n[4] Attempting to load SAM-2 model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"\nConfig path: {config_found}")
    print(f"Model path: {model_found}")
    
    # Try different methods
    methods_to_try = [
        ("Absolute paths", lambda: build_sam2(config_found, model_found, device=device)),
        ("Relative config path", lambda: build_sam2(str(Path(config_found).relative_to(Path.cwd())), model_found, device=device)),
        ("String paths (no absolute)", lambda: build_sam2(str(Path(config_found)), str(Path(model_found)), device=device)),
    ]
    
    model = None
    last_error = None
    
    for method_name, method_func in methods_to_try:
        print(f"\nTrying method: {method_name}...")
        try:
            model = method_func()
            print(f"✓ Success with method: {method_name}")
            break
        except Exception as e:
            print(f"✗ Failed: {str(e)[:100]}...")
            last_error = e
            continue
    
    if model is None:
        print(f"\n✗ All methods failed. Last error:")
        print(f"Error type: {type(last_error).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("Troubleshooting suggestions:")
        print("=" * 60)
        print("1. Check if config file is valid YAML:")
        print(f"   python -c \"import yaml; yaml.safe_load(open('{config_found}'))\"")
        print("\n2. Verify model file is not corrupted:")
        print(f"   Check file size: {Path(model_found).stat().st_size / (1024**2):.1f} MB")
        print("\n3. Try using config from SAM-2 installation directly")
        print("\n4. Check SAM-2 version compatibility")
        return 1
    
    # Test predictor
    print("\n[5] Testing predictor...")
    try:
        predictor = SAM2ImagePredictor(model)
        print("✓ Predictor created successfully!")
    except Exception as e:
        print(f"✗ Failed to create predictor: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! SAM-2 is working correctly.")
    print("=" * 60)
    print(f"\nWorking configuration:")
    print(f"  Config: {config_found}")
    print(f"  Model: {model_found}")
    print(f"  Device: {device}")
    return 0

if __name__ == '__main__':
    sys.exit(test_sam2_loading())

