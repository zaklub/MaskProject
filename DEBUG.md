# Debugging Guide for Remote Machine

## Common Issues and Solutions

### Issue 1: ModuleNotFoundError: No module named 'sam2'

**Check if SAM-2 is installed:**
```bash
python -c "import sam2; print('SAM-2 is installed')"
```

**If not installed, install it:**
```bash
# Option 1: Install from GitHub (Recommended)
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Option 2: Try pip install (may not work)
pip install sam2

# Option 3: Clone and install manually
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..
```

**Verify installation:**
```bash
python -c "import sam2; print(sam2.__file__)"
```

### Issue 2: Check Python Environment

**Verify you're using the correct Python:**
```bash
# Check Python version
python --version

# Check which Python is being used
which python  # Linux/Mac
where python  # Windows

# Check if virtual environment is activated
echo $VIRTUAL_ENV  # Linux/Mac
echo %VIRTUAL_ENV%  # Windows
```

**Activate virtual environment if needed:**
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Issue 3: Check All Dependencies

**Verify all required packages:**
```bash
python -c "
import sys
packages = ['torch', 'ultralytics', 'cv2', 'numpy', 'sam2']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} - MISSING')
        missing.append(pkg)
if missing:
    print(f'\nMissing packages: {missing}')
    sys.exit(1)
else:
    print('\nAll packages installed!')
"
```

### Issue 4: Check Model Files

**Verify model files exist:**
```bash
# Check if model files are present
ls -lh *.pt *.yaml  # Linux/Mac
dir *.pt *.yaml     # Windows

# Expected files:
# - yolov8n.pt (~6 MB)
# - sam2_hiera_large.pt (~1-2 GB)
# - sam2_hiera_l.yaml (small)
```

**If missing, download them:**
```bash
python download_sam2_models.py
```

### Issue 5: Test Script Step by Step

**Test individual components:**
```bash
# Test 1: Import all modules
python -c "
import torch
import ultralytics
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
print('✓ All imports successful')
"

# Test 2: Check CUDA availability (if using GPU)
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Test 3: Test YOLO
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('✓ YOLO loaded successfully')
"

# Test 4: Test SAM-2 (if model file exists)
python -c "
from sam2.build_sam import build_sam2
import os
if os.path.exists('sam2_hiera_large.pt') and os.path.exists('sam2_hiera_l.yaml'):
    model = build_sam2('sam2_hiera_l.yaml', 'sam2_hiera_large.pt', device='cpu')
    print('✓ SAM-2 loaded successfully')
else:
    print('✗ SAM-2 model files not found')
"
```

## Quick Debug Script

Create a file `test_installation.py`:

```python
#!/usr/bin/env python3
"""Quick test script to verify installation"""

import sys

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
    import os
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
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

**Run the test script:**
```bash
python test_installation.py
```

## Remote Machine Specific Tips

### 1. SSH into Remote Machine
```bash
ssh user@remote-machine-ip
```

### 2. Check Python Path
```bash
# Find where Python packages are installed
python -c "import sys; print('\n'.join(sys.path))"
```

### 3. Install in User Space (if no sudo)
```bash
pip install --user git+https://github.com/facebookresearch/segment-anything-2.git
```

### 4. Check Disk Space (for large model files)
```bash
df -h .  # Linux/Mac
dir      # Windows
```

### 5. Check Internet Connection (for downloads)
```bash
ping github.com
curl -I https://github.com
```

## Common Error Messages and Fixes

### "No module named 'sam2'"
**Fix:** Install SAM-2 (see Issue 1 above)

### "CUDA out of memory"
**Fix:** Use CPU mode or reduce GPU memory fraction
```bash
python truck_mask_cleaner.py --image img.jpg --device cpu
# or
python truck_mask_cleaner.py --image img.jpg --device cuda --gpu-memory-fraction 0.5
```

### "FileNotFoundError: sam2_hiera_large.pt"
**Fix:** Download model files
```bash
python download_sam2_models.py
```

### "Permission denied"
**Fix:** Check file permissions or use virtual environment
```bash
chmod +x truck_mask_cleaner.py
# or activate venv
source venv/bin/activate
```

## Getting Help

If issues persist, collect this information:
```bash
# System info
python --version
pip --version
uname -a  # Linux/Mac
systeminfo  # Windows

# Package versions
pip list | grep -E "(torch|ultralytics|opencv|numpy|sam)"

# Error traceback
python truck_mask_cleaner.py --image test.jpg 2>&1 | tee error.log
```

