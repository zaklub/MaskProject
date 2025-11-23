# Testing Guide

## Prerequisites
✅ All model files are in place:
- `yolov8n.pt` - YOLOv8 model
- `sam2_hiera_large.pt` - SAM-2 model checkpoint
- `sam2_hiera_l.yaml` - SAM-2 config file

## Running the Script

### Basic Usage
```powershell
python truck_mask_cleaner.py --image path/to/your/image.jpg
```

### Example with all options
```powershell
python truck_mask_cleaner.py --image truck_xray.jpg --conf-threshold 0.3 --feather-kernel 61
```

## Expected Output Files

After running, you should see three output files in the current directory:
1. **mask.png** - Binary mask showing the detected truck region
2. **background_cleaned.png** - The cleaned background (with noise/streaks removed)
3. **final.png** - Final recombined image with truck and cleaned background

## Testing Steps

1. **Prepare your X-ray image** with a truck visible
   - Supported formats: JPG, PNG, etc. (any format OpenCV supports)
   - Place the image in your project directory or provide full path

2. **Run the script**:
   ```powershell
   python truck_mask_cleaner.py --image your_image.jpg
   ```

3. **Check the output**:
   - The script will print progress messages for each step
   - Check the three output images to verify results

## Troubleshooting

### If YOLO doesn't detect a truck:
- Lower the confidence threshold: `--conf-threshold 0.1`
- Check that the image actually contains a truck
- YOLO looks for COCO class 7 (truck)

### If you get import errors:
- Make sure your virtual environment is activated: `venv\Scripts\activate`
- Verify all packages are installed: `pip list`

### If SAM-2 fails to load:
- Check that `sam2_hiera_large.pt` exists and is the correct file
- Verify `sam2_hiera_l.yaml` is in the current directory
- Try specifying full paths: `--sam2-model path/to/sam2_hiera_large.pt --sam2-config path/to/sam2_hiera_l.yaml`

## Performance Notes

- **CPU-only processing** - This will be slower than GPU but ensures compatibility
- **First run** may take longer as models are loaded into memory
- **Large images** will take more time to process
- **SAM-2 LARGE model** is the most accurate but also the slowest

## Example Test Command

```powershell
# Make sure you're in the project directory
cd C:\AI\ZATCA\YOLO-SAM

# Activate virtual environment (if not already active)
.\venv\Scripts\Activate.ps1

# Run the script
python truck_mask_cleaner.py --image your_truck_xray.jpg
```

