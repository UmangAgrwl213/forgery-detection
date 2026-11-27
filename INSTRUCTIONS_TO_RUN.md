# How to Run the Comprehensive Visualization

## Prerequisites Check

Before running, make sure you have:

1. **Trained model**: `outputs/best_model.pth` exists
2. **Dataset**: `dataset/CASIA2/Tp/` directory with images
3. **Python environment**: All required packages installed

## Run Commands

### Option 1: Simple Demo (Single Image)
```bash
python simple_comprehensive_demo.py
```

### Option 2: Gallery (Multiple Images)  
```bash
python create_comprehensive_gallery.py
```

## What Will Happen

When you run either command:

1. **Model loading**: Loads the trained ResNet34-UNet model
2. **Image processing**: Processes a sample image from your dataset
3. **Prediction**: Runs forgery detection on the image
4. **Visualization**: Creates 4-panel comprehensive view:
   - Original image (top-left)
   - Probability heatmap (top-right) 
   - Binary mask (bottom-left)
   - Detection overlay (bottom-right)
5. **Output**: Saves results to `outputs/comprehensive_demo.png` or `outputs/comprehensive_gallery/`

## Expected Output Files

- `outputs/comprehensive_demo.png` - Single comprehensive visualization
- `outputs/comprehensive_gallery/` - Directory with multiple gallery images

## Troubleshooting

If you encounter errors:

1. **"Model not found"**: Ensure `outputs/best_model.pth` exists
2. **"Dataset not found"**: Ensure `dataset/CASIA2/Tp/` directory exists with images  
3. **"CUDA error"**: The system will automatically fall back to CPU

## View Results

Open the generated PNG files in any image viewer to see:
- Red highlights showing detected forgeries
- Color-coded probability maps
- Clear binary segmentation masks