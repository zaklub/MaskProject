#!/usr/bin/env python3
"""
Download all required model files:
- YOLOv8n model (yolov8n.pt)
- SAM-2 Hiera LARGE model (sam2_hiera_large.pt)
- SAM-2 config file (sam2_hiera_l.yaml)
"""

import os
import urllib.request
from pathlib import Path

def download_file(url, filename):
    """Download a file from URL with progress indicator."""
    print(f"Downloading {filename}...")
    print(f"URL: {url}")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end='')
    
    try:
        urllib.request.urlretrieve(url, filename, show_progress)
        print(f"\n✓ Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading {filename}: {e}")
        return False

def main():
    print("=" * 60)
    print("Model Files Downloader")
    print("=" * 60)
    
    files_to_download = []
    
    # YOLOv8n model
    yolo_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    yolo_filename = "yolov8n.pt"
    files_to_download.append((yolo_url, yolo_filename, "YOLOv8n model"))
    
    # SAM-2 model checkpoint
    sam2_model_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    sam2_model_filename = "sam2_hiera_large.pt"
    files_to_download.append((sam2_model_url, sam2_model_filename, "SAM-2 Hiera LARGE model"))
    
    # SAM-2 config file
    sam2_config_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/configs/sam2_hiera_l.yaml"
    sam2_config_filename = "sam2_hiera_l.yaml"
    files_to_download.append((sam2_config_url, sam2_config_filename, "SAM-2 config file"))
    
    print("\nThis script will download:")
    for url, filename, description in files_to_download:
        print(f"  - {filename} ({description})")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    print()
    
    for url, filename, description in files_to_download:
        # Check if file already exists
        if os.path.exists(filename):
            response = input(f"{filename} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print(f"Skipping {filename}\n")
                continue
            else:
                os.remove(filename)
        
        download_file(url, filename)
        print()
    
    print("=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nAll files saved in: {os.getcwd()}")
    for _, filename, _ in files_to_download:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename} (not downloaded)")

if __name__ == '__main__':
    main()

