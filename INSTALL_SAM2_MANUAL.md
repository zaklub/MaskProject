# Manual SAM-2 Installation Guide

If you cannot clone the repository via git, you can download and install SAM-2 manually.

## Method 1: Download ZIP File

### Step 1: Download the Repository

1. Go to: https://github.com/facebookresearch/segment-anything-2
2. Click the green **"Code"** button
3. Select **"Download ZIP"**
4. Save the file (e.g., `segment-anything-2-main.zip`)

### Step 2: Extract and Install

**On Windows:**
```powershell
# Extract the ZIP file to a folder
# Then navigate to the extracted folder
cd path\to\segment-anything-2-main

# Install in development mode
pip install -e .
```

**On Linux/Mac:**
```bash
# Extract the ZIP file
unzip segment-anything-2-main.zip
cd segment-anything-2-main

# Install in development mode
pip install -e .
```

### Step 3: Verify Installation

```bash
python -c "import sam2; print('SAM-2 installed successfully!')"
```

## Method 2: Download Specific Files Only

If you only need the Python package files:

### Step 1: Create Directory Structure

```bash
# Create a temporary directory
mkdir sam2_manual_install
cd sam2_manual_install
```

### Step 2: Download Required Files

You need to download these files from the repository:
- `setup.py` - Installation script
- `pyproject.toml` - Project configuration (if exists)
- `sam2/` directory - The main package

**Download links:**
- Setup file: https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/setup.py
- Main package: Download the entire `sam2/` folder from the repository

### Step 3: Install

```bash
pip install -e .
```

## Method 3: Copy from Another Machine

If you have SAM-2 installed on another machine:

### Step 1: Find SAM-2 Installation Location

**On the machine where SAM-2 is installed:**
```bash
python -c "import sam2; import os; print(os.path.dirname(sam2.__file__))"
```

### Step 2: Copy the SAM-2 Package

Copy the entire `sam2` directory to your remote machine.

### Step 3: Add to Python Path

**Option A: Install as package**
```bash
# Place sam2 folder in your project directory
# Then install:
pip install -e .  # if setup.py exists in parent directory
```

**Option B: Add to PYTHONPATH**
```bash
# Add to environment variable
export PYTHONPATH="${PYTHONPATH}:/path/to/sam2/parent/directory"

# Or in Python script:
import sys
sys.path.append('/path/to/sam2/parent/directory')
```

## Method 4: Use pip with Local File

### Step 1: Download Repository as ZIP

Download: https://github.com/facebookresearch/segment-anything-2/archive/refs/heads/main.zip

### Step 2: Install from Local File

```bash
# Extract ZIP first, then:
pip install /path/to/segment-anything-2-main/

# Or if you have the ZIP file:
pip install segment-anything-2-main.zip
```

## Troubleshooting

### Issue: "No module named 'sam2' after manual install"

**Solution:**
```bash
# Verify installation location
python -c "import sys; print('\n'.join(sys.path))"

# Check if sam2 is in site-packages
python -c "import sam2; print(sam2.__file__)"
```

### Issue: "Missing dependencies"

**Solution:** Install required dependencies first:
```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install Pillow
pip install omegaconf
pip install hydra-core
```

### Issue: "Cannot find setup.py"

**Solution:** The repository might use a different build system. Try:
```bash
# Check for pyproject.toml
ls pyproject.toml

# If exists, install with:
pip install .
```

## Quick Installation Script

Create a file `install_sam2_manual.py`:

```python
#!/usr/bin/env python3
"""Helper script to install SAM-2 from downloaded ZIP"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

def install_from_zip(zip_path):
    """Install SAM-2 from downloaded ZIP file"""
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        print(f"Error: ZIP file not found: {zip_path}")
        return False
    
    # Extract to temporary directory
    extract_dir = zip_path.parent / "sam2_extract"
    extract_dir.mkdir(exist_ok=True)
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Find the extracted folder
    extracted_folders = list(extract_dir.iterdir())
    if not extracted_folders:
        print("Error: ZIP file appears to be empty")
        return False
    
    sam2_dir = extracted_folders[0]  # Usually segment-anything-2-main or similar
    
    print(f"Installing from {sam2_dir}...")
    os.chdir(sam2_dir)
    
    # Install
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])
    
    if result.returncode == 0:
        print("✓ SAM-2 installed successfully!")
        return True
    else:
        print("✗ Installation failed")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python install_sam2_manual.py <path_to_zip_file>")
        print("Example: python install_sam2_manual.py segment-anything-2-main.zip")
        sys.exit(1)
    
    zip_path = sys.argv[1]
    success = install_from_zip(zip_path)
    sys.exit(0 if success else 1)
```

**Usage:**
```bash
# Download the ZIP file first, then:
python install_sam2_manual.py segment-anything-2-main.zip
```

## Recommended Approach

**Easiest method:**
1. Download ZIP from GitHub
2. Extract it
3. Navigate to extracted folder
4. Run: `pip install -e .`

This is the most reliable method when git clone is not available.

