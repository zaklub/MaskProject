# GPU Installation Guide for NVIDIA Quadro P1000 (4GB VRAM)

This guide covers installing and running the truck mask generator on a machine with NVIDIA Quadro P1000 GPU (4GB VRAM).

## System Requirements

- **GPU**: NVIDIA Quadro P1000 (4GB VRAM) or compatible
- **CUDA**: Version 11.8 or 12.1 (check your driver version)
- **Python**: 3.8 or higher
- **OS**: Windows 10/11, Linux, or macOS (with compatible GPU)

## Step 1: Check CUDA Installation

First, verify CUDA is installed and accessible:

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version (if installed)
nvcc --version
```

**Expected output from nvidia-smi:**
- Should show your Quadro P1000
- CUDA Version should be displayed (e.g., 12.1, 11.8)

## Step 2: Install CUDA Toolkit (if not installed)

### For Windows:
1. Download CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads
2. For Quadro P1000, use **CUDA 11.8** or **CUDA 12.1**
3. Run the installer and follow the prompts
4. Restart your computer

### For Linux:
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

## Step 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

## Step 4: Install PyTorch with CUDA Support

**Important**: Choose the correct CUDA version for your system.

### For CUDA 11.8 (Recommended for Quadro P1000):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### For CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify PyTorch GPU Support:
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output:**
```
CUDA available: True
CUDA device: Quadro P1000
```

## Step 5: Install Other Dependencies

```bash
# Install remaining packages
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install Pillow>=9.5.0
pip install matplotlib>=3.7.0
```

## Step 6: Install SAM-2

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

## Step 7: Download Model Files

Run the download script:
```bash
python download_sam2_models.py
```

Or download manually:
- `yolov8n.pt` - Auto-downloads on first use
- `sam2_hiera_large.pt` - Download from SAM-2 repository
- `sam2_hiera_l.yaml` - Config file (copied from venv or downloaded)

## Step 8: Test GPU Installation

```bash
# Test the script with GPU
python truck_mask_cleaner.py --image your_image.jpg --device cuda
```

## Memory Optimization for 4GB VRAM

The Quadro P1000 has 4GB VRAM, which is limited. The script automatically:
- Uses 80% of GPU memory by default
- Can be adjusted with `--gpu-memory-fraction`

### Recommended Settings for 4GB GPU:

```bash
# Use 70% of GPU memory (safer for 4GB)
python truck_mask_cleaner.py --image your_image.jpg --device cuda --gpu-memory-fraction 0.7

# Use 60% if you encounter out-of-memory errors
python truck_mask_cleaner.py --image your_image.jpg --device cuda --gpu-memory-fraction 0.6
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce GPU memory fraction:
```bash
python truck_mask_cleaner.py --image your_image.jpg --device cuda --gpu-memory-fraction 0.5
```

### Issue: "CUDA not available"
**Solutions**:
1. Check NVIDIA driver: `nvidia-smi`
2. Verify PyTorch CUDA installation (see Step 4)
3. Restart your computer after CUDA installation
4. Check CUDA version compatibility

### Issue: "No CUDA-capable device is detected"
**Solutions**:
1. Ensure GPU drivers are installed
2. Check GPU is recognized: `nvidia-smi`
3. Verify GPU supports CUDA (Quadro P1000 does support CUDA)

### Issue: Slow performance on GPU
**Solutions**:
1. Ensure you're using `--device cuda` (not `auto`)
2. Check GPU utilization: `nvidia-smi` while running
3. Close other GPU-intensive applications
4. Consider using smaller models if needed

## Performance Comparison

Expected performance on Quadro P1000 (4GB):
- **YOLOv8n detection**: ~50-100ms (GPU) vs ~200-500ms (CPU)
- **SAM-2 segmentation**: ~2-5 seconds (GPU) vs ~10-30 seconds (CPU)

## Quick Start Command

```bash
# Activate venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Run with GPU
python truck_mask_cleaner.py --image your_image.jpg --device cuda --gpu-memory-fraction 0.7
```

## Notes

- The script automatically detects GPU if `--device auto` is used (default)
- For 4GB VRAM, use `--gpu-memory-fraction 0.6-0.7` to avoid out-of-memory errors
- SAM-2 LARGE model is memory-intensive; if issues persist, consider using a smaller SAM-2 model

