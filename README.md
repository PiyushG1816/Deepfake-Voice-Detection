# AASIST-Based Speech Deepfake Detection

## Overview
This project implements a speech deepfake detection system using the AASIST model architecture. The model is trained on MFCC features extracted from audio files and can classify input audio as real or fake.

## Features
- MFCC-based preprocessing
- AASIST-inspired CNN + Attention model
- Early stopping and best model checkpointing
- Evaluation on custom dataset

## Project Structure
```
.
├── Model.py            # Model definition
├── Load_data.py        # Loads and saves the data as .csv
├── Preprocess.py       # Audio preprocessing (MFCC + multiprocessing)
├── Requirements.txt    # Required Python libraries
├── Model.pt            # Saved PyTorch model (optional)
├── REPORT.md           # Project documentation 
└── README.md           # This file
```

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/deepfake-speech-detector.git
cd deepfake-speech-detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset Preparation
Organize your dataset as follows:
```
data_root/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
```

### 4. Preprocess Audio
```bash
python Preprocess.py
```
This generates:
- `train_features.npy`, `train_labels.npy`
- `val_features.npy`, `val_labels.npy`

### 5. Train the Model
```bash
python Model.py
```

## Results
- **Validation Accuracy**: ~86.91%
- **Early Stopping**: Enabled to avoid overfitting

## Model Architecture (Simplified)
- 2D CNN Layers
- Multi-head Self-Attention
- Global Pooling
- Fully Connected Layers


## Acknowledgements
- [AASIST Paper](https://arxiv.org/abs/2110.01200)
- PyTorch Team
- Librosa for audio processing

