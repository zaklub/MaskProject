# Installation Guide for Truck Mask Cleaner

## GPU vs CPU Installation

- **For CPU-only machines**: Follow this guide
- **For NVIDIA GPU (e.g., Quadro P1000)**: See [INSTALL_GPU.md](INSTALL_GPU.md) for GPU-specific instructions

## Required Libraries

Install all dependencies using pip:

```bash
pip install -r requirements.txt
```

## Individual Package Installation

If you prefer to install packages individually:

### 1. PyTorch (CPU-only)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Ultralytics (YOLOv8)
```bash
pip install ultralytics
```

### 3. OpenCV
```bash
pip install opencv-python
```

### 4. NumPy
```bash
pip install numpy
```

### 5. Pillow
```bash
pip install Pillow
```

### 6. SAM-2 (Segment Anything 2)

**Important**: SAM-2 installation can be more complex. Choose one of the following methods:

#### Option A: Install from GitHub (Recommended - if git is available)
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

#### Option B: Manual Installation from ZIP (If git clone doesn't work)
1. Download ZIP from: https://github.com/facebookresearch/segment-anything-2
   - Click "Code" → "Download ZIP"
2. Extract the ZIP file
3. Install:
```bash
cd segment-anything-2-main  # or the extracted folder name
pip install -e .
```

**Or use the helper script:**
```bash
# After downloading the ZIP file:
python install_sam2_manual.py segment-anything-2-main.zip
```

#### Option C: Clone and Install (if git is available)
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
```

#### Option D: Install via pip (if available)
```bash
pip install sam2
```

**Note**: 
- If you cannot use git, see [INSTALL_SAM2_MANUAL.md](INSTALL_SAM2_MANUAL.md) for detailed manual installation instructions
- After installing SAM-2, you'll need to download the model files (see below)

## Model Files Required

You'll need to download these model files:

1. **YOLOv8n model**: Download using one of the methods below:

### Method 1: Use the Download Script (Easiest)
```bash
python download_sam2_models.py
```
This script downloads all three files (YOLOv8n, SAM-2 model, and config).

### Method 2: Direct Download URLs

**YOLOv8n Model** (~6 MB):
- Direct download: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
- Or use PowerShell:
  ```powershell
  Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt" -OutFile "yolov8n.pt"
  ```

**Note**: YOLOv8 will also auto-download this model on first use if it's not found, but downloading it manually ensures it's available offline.

2. **SAM-2 Hiera LARGE model**: Download using one of the methods below:

### Method 1: Use the Download Script (Easiest)
```bash
python download_sam2_models.py
```

### Method 2: Direct Download URLs

**Model Checkpoint** (~1-2 GB):
- Direct download: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
- Or use PowerShell:
  ```powershell
  Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt" -OutFile "sam2_hiera_large.pt"
  ```

**Config File**:
- Direct download: https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/configs/sam2_hiera_l.yaml
- Or use PowerShell:
  ```powershell
  Invoke-WebRequest -Uri "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/configs/sam2_hiera_l.yaml" -OutFile "sam2_hiera_l.yaml"
  ```

### Method 3: Using Hugging Face Hub
```python
from huggingface_hub import hf_hub_download

# Download model checkpoint
hf_hub_download(repo_id="facebook/sam2-hiera-large", filename="sam2_hiera_large.pt", local_dir="./")

# Config file still needs to be downloaded from GitHub (see Method 2)
```

**Place both files** (`sam2_hiera_large.pt` and `sam2_hiera_l.yaml`) in your working directory where you'll run the script.

## Verify Installation

Test that all packages are installed correctly:

```bash
python -c "import torch; import ultralytics; import cv2; import numpy; print('All packages installed successfully!')"
```

## Troubleshooting

### If SAM-2 installation fails:
- Make sure you have a C++ compiler installed (required for some dependencies)
- On Windows: Install Visual Studio Build Tools
- On Linux: `sudo apt-get install build-essential`
- On macOS: `xcode-select --install`

### If PyTorch installation fails:
- Visit [PyTorch website](https://pytorch.org/) to get the correct installation command for your system
- Make sure you're installing the CPU-only version if you don't have CUDA

### If you encounter import errors:
- Make sure you're using Python 3.8 or higher
- Try creating a fresh virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

