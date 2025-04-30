import torch
import torch.nn as nn
import numpy as np
import librosa
import os

# Define the AASIST model again (same as your model.py)
class AASIST(nn.Module):
    def __init__(self, num_classes=2):
        super(AASIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x, _ = self.attn(x, x, x)
        x = self.global_pool(x.permute(0, 2, 1).unsqueeze(-1))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AASIST(num_classes=2).to(device)
model.load_state_dict(torch.load('Model.pt', map_location=device))
model.eval()

def extract_mfcc(file_path):
    try:
        # Load audio
        waveform, sr = librosa.load(file_path, sr=16000, mono=True)

        # Check if empty after trimming
        waveform, _ = librosa.effects.trim(waveform)
        if waveform.size == 0 or np.max(np.abs(waveform)) < 0.01:  # Threshold for silence
            print(f"Silent audio detected: {file_path}")
            return None

        # Normalize
        waveform = librosa.util.normalize(waveform)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)

        # Pad or truncate MFCCs
        MAX_MFCC_LENGTH = 200
        if mfcc.shape[1] < MAX_MFCC_LENGTH:
            pad_width = MAX_MFCC_LENGTH - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_MFCC_LENGTH]

        return mfcc.astype(np.float32)

    except Exception as e:
        print(f"Error in extract_mfcc: {e}")
        return None

def predict_audio(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == '.npy':
        mfcc = np.load(file_path)
    elif ext in ['.wav', '.mp3']:
        mfcc = extract_mfcc(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a .npy, .wav, or .mp3 file.")
    
    # Handle silent audio case
    if mfcc is None:
        return "silent"  # Return string flag for silent audio
    
    # Ensure we have a valid array
    if not isinstance(mfcc, np.ndarray):
        return "error"
    
    mfcc = torch.tensor(mfcc).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, freq, time]

    with torch.no_grad():
        output = model(mfcc)
        _, predicted = torch.max(output, 1)
        label = predicted.item()

    return 'Real' if label == 0 else 'Fake'

