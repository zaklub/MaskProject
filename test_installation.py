#!/usr/bin/env python3
"""Quick test script to verify installation"""

import sys
import os

def test_import(module_name, import_statement=None):
    """Test if a module can be imported"""
    if import_statement is None:
        import_statement = f"import {module_name}"
    try:
        exec(import_statement)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name} - {e}")
        return False
    except Exception as e:
        print(f"✗ {module_name} - {e}")
        return False

def main():
    print("=" * 60)
    print("Testing Installation")
    print("=" * 60)
    
    all_ok = True
    
    # Test basic packages
    print("\n[1] Testing basic packages...")
    all_ok &= test_import("torch")
    all_ok &= test_import("cv2")
    all_ok &= test_import("numpy")
    all_ok &= test_import("PIL", "from PIL import Image")
    
    # Test YOLO
    print("\n[2] Testing YOLO...")
    all_ok &= test_import("ultralytics", "from ultralytics import YOLO")
    
    # Test SAM-2
    print("\n[3] Testing SAM-2...")
    all_ok &= test_import("sam2", "import sam2")
    all_ok &= test_import("sam2.build_sam", "from sam2.build_sam import build_sam2")
    all_ok &= test_import("sam2.sam2_image_predictor", "from sam2.sam2_image_predictor import SAM2ImagePredictor")
    
    # Test model files
    print("\n[4] Testing model files...")
    files = {
        "yolov8n.pt": "YOLOv8 model",
        "sam2_hiera_large.pt": "SAM-2 model",
        "sam2_hiera_l.yaml": "SAM-2 config"
    }
    for filename, desc in files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024 * 1024)
            print(f"✓ {filename} ({desc}) - {size:.1f} MB")
        else:
            print(f"✗ {filename} ({desc}) - NOT FOUND")
            all_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All tests passed! Installation looks good.")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nTo install SAM-2, run:")
        print("  pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        return 1

if __name__ == '__main__':
    sys.exit(main())

