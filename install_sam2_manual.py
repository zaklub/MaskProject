#!/usr/bin/env python3
"""Helper script to install SAM-2 from downloaded ZIP file"""

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
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except Exception as e:
        print(f"Error extracting ZIP: {e}")
        return False
    
    # Find the extracted folder
    extracted_folders = [f for f in extract_dir.iterdir() if f.is_dir()]
    if not extracted_folders:
        print("Error: ZIP file appears to be empty or no directories found")
        return False
    
    sam2_dir = extracted_folders[0]  # Usually segment-anything-2-main or similar
    print(f"Found extracted folder: {sam2_dir}")
    
    # Check for setup.py or pyproject.toml
    if not (sam2_dir / "setup.py").exists() and not (sam2_dir / "pyproject.toml").exists():
        print("Warning: No setup.py or pyproject.toml found. Trying to install anyway...")
    
    print(f"Installing from {sam2_dir}...")
    original_dir = os.getcwd()
    
    try:
        os.chdir(sam2_dir)
        
        # Install
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ SAM-2 installed successfully!")
            print(result.stdout)
            return True
        else:
            print("✗ Installation failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    finally:
        os.chdir(original_dir)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("=" * 60)
        print("SAM-2 Manual Installation Helper")
        print("=" * 60)
        print("\nUsage: python install_sam2_manual.py <path_to_zip_file>")
        print("\nExample:")
        print("  python install_sam2_manual.py segment-anything-2-main.zip")
        print("\nSteps:")
        print("  1. Download ZIP from: https://github.com/facebookresearch/segment-anything-2")
        print("  2. Click 'Code' -> 'Download ZIP'")
        print("  3. Run this script with the ZIP file path")
        sys.exit(1)
    
    zip_path = sys.argv[1]
    success = install_from_zip(zip_path)
    sys.exit(0 if success else 1)

