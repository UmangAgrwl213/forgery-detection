# Comprehensive Forgery Detection Visualization

This project demonstrates how to visualize forgery detection results with all components shown together.

## What You'll See in the Visualization

The comprehensive visualization shows four panels side-by-side:

1. **Top-left**: Original image
2. **Top-right**: Probability heatmap (blue = low probability, red = high probability)
3. **Bottom-left**: Binary mask (white = detected forgery regions)
4. **Bottom-right**: Detection overlay (red highlights for detected forgeries)

## How to Run

```bash
# Run the comprehensive demo
python simple_comprehensive_demo.py

# Or run the gallery creation script
python create_comprehensive_gallery.py
```

## Output Files

After running, you'll find:
- `outputs/comprehensive_demo.png` - Single image with all components
- `outputs/comprehensive_gallery/` - Directory with multiple gallery images

## Understanding the Results

### Probability Heatmap
- Blue areas: Low probability of forgery (authentic regions)
- Red areas: High probability of forgery (tampered regions)

### Binary Mask
- White areas: Detected forgery regions
- Black areas: Authentic regions

### Detection Overlay
- Original image with red highlights showing detected forgeries
- Red color indicates areas where the model detected tampering

## Requirements

Make sure you have:
1. Trained model in `outputs/best_model.pth`
2. CASIA v2 dataset in `dataset/CASIA2/Tp/`
3. All dependencies installed (torch, opencv-python, numpy)

## Example Output

The visualization will show exactly what you requested - the real image, probability heatmap, binary mask, and detection overlay all together in one comprehensive view that clearly demonstrates:
- Where forgeries are located
- How confident the model is about each detection
- The distinction between authentic and tampered regions