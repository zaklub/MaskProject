#!/usr/bin/env python3
"""
Truck Mask Generator - X-ray Image Processing Script
Detects trucks in X-ray images and generates pixel-perfect masks.
Uses YOLOv8n for detection and SAM-2 (hiera LARGE) for segmentation.
Supports both CPU and GPU (CUDA) execution with automatic device detection.
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch


def load_image(image_path):
    """Load image from file path."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    # Convert BGR to RGB for processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, img  # Return both RGB and BGR versions


def detect_truck_yolo(image_rgb, model_path='yolov8n.pt', conf_threshold=0.25, device='auto'):
    """
    Detect truck using YOLOv8n model.
    Returns bounding box (x1, y1, x2, y2) of most confident detection.
    
    Args:
        device: 'auto', 'cpu', or 'cuda'. If 'auto', detects best available device.
    """
    # Auto-detect device if needed
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load YOLOv8n model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_rgb, device=device, verbose=False)
    
    # COCO class ID: 7=truck
    truck_class_id = 7
    
    # Get class names from model
    class_names = model.names
    
    detections = []
    all_detections = []  # For debugging
    
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Store all detections for debugging
                class_name = class_names.get(cls, f"class_{cls}")
                all_detections.append((cls, class_name, conf, x1, y1, x2, y2))
                
                # Filter for truck only
                if cls == truck_class_id and conf >= conf_threshold:
                    detections.append((x1, y1, x2, y2, conf))
    
    # If no truck detected, show what was detected
    if not detections:
        print("\n⚠️  No truck detected. All detections above threshold:")
        if all_detections:
            # Sort by confidence
            all_detections.sort(key=lambda x: x[2], reverse=True)
            for cls, class_name, conf, x1, y1, x2, y2 in all_detections:
                if conf >= conf_threshold * 0.5:  # Show detections above half threshold
                    print(f"  - {class_name} (class {cls}): confidence {conf:.3f}, bbox: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
        else:
            print("  No objects detected at all.")
        
        print(f"\n💡 Suggestions:")
        print(f"  - Lower confidence threshold: --conf-threshold 0.1")
        print(f"  - Check if the image contains a visible truck")
        print(f"  - X-ray images may need lower thresholds as YOLO was trained on natural images")
        
        raise ValueError("YOLO failed to detect a truck in the image. Please check the image or adjust confidence threshold.")
    
    # Select most confident detection
    detections.sort(key=lambda x: x[4], reverse=True)
    x1, y1, x2, y2, conf = detections[0]
    
    print(f"Truck detected with confidence: {conf:.3f}")
    print(f"Bounding box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
    
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def segment_truck_sam2(image_rgb, bbox, sam2_model_path='sam2_hiera_large.pt', sam2_config_path='sam2_hiera_l.yaml', device='auto'):
    """
    Generate pixel-perfect mask using SAM-2 (hiera LARGE model).
    Returns binary mask (uint8, 0-255).
    
    Args:
        device: 'auto', 'cpu', or 'cuda'. If 'auto', detects best available device.
    """
    # Auto-detect device if needed
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Verify config file exists and find the best one
    config_path = Path(sam2_config_path)
    
    # Always try to find config in SAM-2 installation (prefer configs/ subdirectory)
    import sam2
    sam2_path = Path(sam2.__file__).parent
    alt_config_paths = [
        sam2_path / 'configs' / 'sam2' / 'sam2_hiera_l.yaml',  # Preferred: from configs subdirectory
        sam2_path / 'configs' / 'sam2.1' / 'sam2.1_hiera_l.yaml',  # Alternative version
        sam2_path / 'sam2_hiera_l.yaml',  # Fallback: from root
    ]
    
    # If provided config doesn't exist, or if we want to prefer the one from configs/
    if not config_path.exists() or not config_path.is_absolute():
        for alt_path in alt_config_paths:
            if alt_path.exists():
                sam2_config_path = str(alt_path)
                print(f"Using config file from SAM-2 installation: {sam2_config_path}")
                break
        else:
            # Last resort: use provided path if it exists
            if config_path.exists():
                sam2_config_path = str(config_path.absolute())
            else:
                raise FileNotFoundError(
                    f"SAM-2 config file not found: {sam2_config_path}\n"
                    f"Searched in:\n" + "\n".join([f"  - {p}" for p in alt_config_paths])
                )
    else:
        # Use provided config, but verify it exists
        if not config_path.exists():
            # Try alternatives
            for alt_path in alt_config_paths:
                if alt_path.exists():
                    sam2_config_path = str(alt_path)
                    print(f"Provided config not found, using: {sam2_config_path}")
                    break
            else:
                raise FileNotFoundError(f"SAM-2 config file not found: {sam2_config_path}")
        else:
            sam2_config_path = str(config_path.absolute())
    
    # Verify model file exists
    model_path = Path(sam2_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"SAM-2 model file not found: {sam2_model_path}\n"
            f"Please download it using: python download_sam2_models.py"
        )
    sam2_model_path = str(model_path.absolute())
    
    # Load SAM-2 model
    print(f"Loading SAM-2 model on {device.upper()}...")
    print(f"Config: {sam2_config_path}")
    print(f"Model: {sam2_model_path}")
    
    # Try loading with different config paths if first attempt fails
    configs_to_try = [sam2_config_path]
    
    # Add alternative config paths from SAM-2 installation
    try:
        import sam2
        sam2_path = Path(sam2.__file__).parent
        alt_configs = [
            sam2_path / 'configs' / 'sam2' / 'sam2_hiera_l.yaml',
            sam2_path / 'configs' / 'sam2.1' / 'sam2.1_hiera_l.yaml',
        ]
        for alt_cfg in alt_configs:
            if alt_cfg.exists() and str(alt_cfg) not in configs_to_try:
                configs_to_try.append(str(alt_cfg))
    except:
        pass
    
    sam2_model = None
    last_error = None
    
    for cfg_to_try in configs_to_try:
        try:
            print(f"Trying config: {cfg_to_try}")
            sam2_model = build_sam2(cfg_to_try, sam2_model_path, device=device)
            print(f"✓ Successfully loaded with config: {cfg_to_try}")
            break
        except Exception as e:
            last_error = e
            print(f"✗ Failed with config {Path(cfg_to_try).name}: {str(e)[:100]}...")
            continue
    
    if sam2_model is None:
        raise RuntimeError(
            f"Failed to load SAM-2 model with all attempted config files.\n"
            f"Last error: {str(last_error)}\n"
            f"Tried configs:\n" + "\n".join([f"  - {c}" for c in configs_to_try]) + "\n"
            f"Model path: {sam2_model_path}\n"
            f"Please check:\n"
            f"  1. Config files exist and are valid YAML\n"
            f"  2. Model file exists and is not corrupted\n"
            f"  3. SAM-2 version is compatible"
        )
    
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Set image for predictor
    predictor.set_image(image_rgb)
    
    # Convert bbox to format expected by SAM-2: [x1, y1, x2, y2]
    box = bbox.astype(np.float32)
    
    # Predict mask
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    
    # Select highest scoring mask
    if len(masks) > 1:
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        print(f"SAM-2 returned {len(masks)} masks, selected mask with score: {scores[best_idx]:.3f}")
    else:
        mask = masks[0]
        print(f"SAM-2 mask score: {scores[0]:.3f}")
    
    # Convert to uint8 binary mask (0 or 255)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    
    return mask_uint8


def main():
    parser = argparse.ArgumentParser(
        description='Detect truck in X-ray image and generate pixel-perfect mask.'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input X-ray image'
    )
    parser.add_argument(
        '--yolo-model',
        type=str,
        default='yolov8n.pt',
        help='Path to YOLOv8 model file (default: yolov8n.pt)'
    )
    parser.add_argument(
        '--sam2-model',
        type=str,
        default='sam2_hiera_large.pt',
        help='Path to SAM-2 model file (default: sam2_hiera_large.pt)'
    )
    parser.add_argument(
        '--sam2-config',
        type=str,
        default='sam2_hiera_l.yaml',
        help='Path to SAM-2 config file (default: sam2_hiera_l.yaml)'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.15,
        help='YOLO confidence threshold (default: 0.15, lower for X-ray images)'
    )
    parser.add_argument(
        '--save-overlay',
        action='store_true',
        help='Also save an overlay image showing mask on original image'
    )
    parser.add_argument(
        '--skip-yolo',
        action='store_true',
        help='Skip YOLO detection and use manual bounding box'
    )
    parser.add_argument(
        '--bbox',
        type=str,
        default=None,
        help='Manual bounding box: "x1,y1,x2,y2" (e.g., "100,200,500,600"). Required if --skip-yolo is used'
    )
    parser.add_argument(
        '--continue-on-yolo-fail',
        action='store_true',
        help='Continue with SAM-2 even if YOLO fails (will use full image or manual bbox)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use: auto (detect), cpu, or cuda (default: auto)'
    )
    parser.add_argument(
        '--gpu-memory-fraction',
        type=float,
        default=0.8,
        help='Fraction of GPU memory to use (0.0-1.0, default: 0.8). Lower for 4GB GPUs.'
    )
    
    args = parser.parse_args()
    
    # Detect and set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            print(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
            
            # Set memory fraction for limited VRAM (e.g., 4GB GPUs)
            if gpu_memory <= 4.5:
                print(f"⚠️  Limited VRAM detected. Using {args.gpu_memory_fraction*100:.0f}% of GPU memory.")
                torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        else:
            device = 'cpu'
            print("No GPU detected. Using CPU.")
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
    
    print(f"Using device: {device.upper()}")
    
    print("=" * 60)
    print("Truck Mask Generator - X-ray Image Processing")
    print("=" * 60)
    
    # Step 1: Load image
    print("\n[Step 1] Loading image...")
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    image_rgb, image_bgr = load_image(image_path)
    print(f"Image loaded: {image_rgb.shape[1]}x{image_rgb.shape[0]} pixels")
    
    # Step 2: Get bounding box (YOLO or manual)
    bbox = None
    
    if args.skip_yolo:
        # Skip YOLO, use manual bounding box
        if args.bbox is None:
            raise ValueError("--bbox is required when using --skip-yolo. Format: 'x1,y1,x2,y2'")
        
        print("\n[Step 2] Using manual bounding box...")
        try:
            coords = [float(x.strip()) for x in args.bbox.split(',')]
            if len(coords) != 4:
                raise ValueError("Bounding box must have 4 values: x1,y1,x2,y2")
            bbox = np.array(coords, dtype=np.float32)
            print(f"Manual bounding box: ({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f})")
        except Exception as e:
            raise ValueError(f"Invalid bounding box format: {args.bbox}. Use format: 'x1,y1,x2,y2'. Error: {e}")
    
    else:
        # Try YOLO detection
        print("\n[Step 2] Detecting truck with YOLOv8n...")
        try:
            bbox = detect_truck_yolo(image_rgb, args.yolo_model, args.conf_threshold, device=device)
        except ValueError as e:
            if args.continue_on_yolo_fail:
                print(f"\n⚠️  YOLO detection failed: {e}")
                print("Continuing with SAM-2...")
                
                # If manual bbox provided, use it
                if args.bbox is not None:
                    print("Using provided manual bounding box...")
                    try:
                        coords = [float(x.strip()) for x in args.bbox.split(',')]
                        if len(coords) != 4:
                            raise ValueError("Bounding box must have 4 values: x1,y1,x2,y2")
                        bbox = np.array(coords, dtype=np.float32)
                        print(f"Manual bounding box: ({bbox[0]:.1f}, {bbox[1]:.1f}) to ({bbox[2]:.1f}, {bbox[3]:.1f})")
                    except Exception as ex:
                        raise ValueError(f"Invalid bounding box format: {args.bbox}. Use format: 'x1,y1,x2,y2'. Error: {ex}")
                else:
                    # Use full image as bounding box
                    print("No manual bbox provided. Using full image as bounding box...")
                    h, w = image_rgb.shape[:2]
                    bbox = np.array([0, 0, w, h], dtype=np.float32)
                    print(f"Full image bounding box: (0, 0) to ({w}, {h})")
            else:
                # Re-raise the error if not continuing
                raise
    
    # Step 3: SAM-2 segmentation
    print("\n[Step 3] Generating pixel-perfect mask with SAM-2...")
    mask = segment_truck_sam2(image_rgb, bbox, args.sam2_model, args.sam2_config, device=device)
    
    # Ensure mask matches image dimensions
    if mask.shape[:2] != image_rgb.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Step 4: Save outputs
    print("\n[Step 4] Saving output images...")
    
    # Convert mask to BGR for saving
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('mask.png', mask_bgr)
    print("✓ mask.png saved")
    
    # Optionally save overlay visualization
    if args.save_overlay:
        # Create overlay: original image with mask highlighted
        overlay = image_bgr.copy()
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        cv2.imwrite('mask_overlay.png', overlay)
        print("✓ mask_overlay.png saved")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

