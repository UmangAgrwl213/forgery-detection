# 🧠 PROJECT MEMORY

[PROJECT]
Name: Forgery Detection
Goal: Detect manipulated regions in images
Status: debugging

[ARCHITECTURE]
Model: UNet
Input: 256x256
Loss: BCE + Dice
Optimizer: Adam

[DATASET]
Source: Kaggle
Issues: possible imbalance, noise

[CURRENT ISSUE]
Validation loss stuck ~0.8

[OBSERVATIONS]
- Train loss decreasing
- Val loss plateau

[HYPOTHESIS]
- Underfitting OR noisy labels

[EXPERIMENTS]
## v1 - baseline
Result: val loss ~0.8