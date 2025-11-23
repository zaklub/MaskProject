# Truck Mask Generator

X-ray image processing tool that detects trucks and generates pixel-perfect masks using YOLOv8 and SAM-2.

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd YOLO-SAM
```

### 2. Install Dependencies

**For CPU-only:**
```bash
pip install -r requirements.txt
```

**For GPU (NVIDIA CUDA):**
```bash
# See INSTALL_GPU.md for detailed instructions
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt
```

### 3. Download Model Files

**Important**: Model files are NOT included in the repository (too large). You need to download them:

```bash
python download_sam2_models.py
```

This will download:
- `yolov8n.pt` (~6 MB) - YOLOv8 nano model
- `sam2_hiera_large.pt` (~1-2 GB) - SAM-2 Hiera LARGE model
- `sam2_hiera_l.yaml` - SAM-2 config file

### 4. Run the Script

```bash
# Basic usage
python truck_mask_cleaner.py --image your_image.jpg

# Use CPU explicitly
python truck_mask_cleaner.py --image your_image.jpg --device cpu

# Use GPU (if CUDA installed)
python truck_mask_cleaner.py --image your_image.jpg --device cuda
```

## What Gets Committed vs Downloaded

### ✅ Committed to Git (Small Files):
- `truck_mask_cleaner.py` - Main script
- `download_sam2_models.py` - Model downloader
- `requirements.txt` - CPU dependencies
- `requirements-gpu.txt` - GPU dependencies
- `INSTALL.md` - CPU installation guide
- `INSTALL_GPU.md` - GPU installation guide
- `sam2_hiera_l.yaml` - Config file (small)
- `.gitignore` - Git ignore rules
- `README.md` - This file

### ❌ NOT Committed (Download Separately):
- `yolov8n.pt` (~6 MB) - Auto-downloads on first YOLO use, or use download script
- `sam2_hiera_large.pt` (~1-2 GB) - Too large for Git, must download
- `venv/` - Virtual environment (recreate locally)
- Output images (`mask.png`, etc.) - Generated files

## Why Not Commit Model Files?

1. **Size**: `sam2_hiera_large.pt` is 1-2 GB (exceeds GitHub's 100MB file limit)
2. **Git LFS**: Would require Git LFS setup (adds complexity)
3. **Best Practice**: Model files are usually downloaded, not versioned
4. **Flexibility**: Users can choose which models to download

## Alternative: Using Git LFS (Optional)

If you want to commit model files, you can use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pt"
git add .gitattributes

# Then commit models
git add *.pt
git commit -m "Add model files via LFS"
```

**Note**: This requires Git LFS to be installed on all machines that clone the repo.

## Project Structure

```
YOLO-SAM/
├── truck_mask_cleaner.py      # Main script
├── download_sam2_models.py    # Model downloader
├── requirements.txt            # CPU dependencies
├── requirements-gpu.txt        # GPU dependencies
├── INSTALL.md                  # CPU installation guide
├── INSTALL_GPU.md              # GPU installation guide
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── sam2_hiera_l.yaml          # SAM-2 config (committed)
├── yolov8n.pt                 # YOLO model (download)
└── sam2_hiera_large.pt        # SAM-2 model (download)
```

## Requirements

- Python 3.8+
- PyTorch (CPU or CUDA)
- See `requirements.txt` or `requirements-gpu.txt` for full list

## Documentation

- **CPU Installation**: See [INSTALL.md](INSTALL.md)
- **GPU Installation**: See [INSTALL_GPU.md](INSTALL_GPU.md)

## License

[Add your license here]

