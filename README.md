# 🔍 Deepfake Detection using AASIST

This repository contains the implementation of a deepfake detection system using the **AASIST (Audio Anti-Spoofing System)** model. The model is trained on MFCC (Mel Frequency Cepstral Coefficient) features extracted from audio data and is designed to classify real vs. spoofed (deepfake) speech.

---

## 📁 Project Structure
 
├── Load_data.py  # Loads the data anda saves it in .csv
├── Model.py # AASIST model definition and training loop 
├── utils/ # Utility functions (augmentation, preprocessing) 
├── Model.pt # Trained model weights 
├── requirements.txt # All Python dependencies 
├── README.md # You're reading it
