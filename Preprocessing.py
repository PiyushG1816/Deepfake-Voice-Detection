import os
import torchaudio.transforms as T
import librosa
import numpy as np
import noisereduce as nr
import multiprocessing
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Dataset directory
dataset_root = r"D:\OneDrive\Desktop\Projects\DeepFake\Speech Dataset\for-norm\for-norm"

# Define splits and labels
splits = ["Training", "Testing", "Validation"]
labels = {"fake": 0, "real": 1}  # Convert labels to numerical format

# Audio parameters
SAMPLE_RATE = 16000  # AASIST requires 16kHz
N_MELS = 128  # Number of Mel filterbanks
N_FFT = 512  # FFT window size
HOP_LENGTH = 160  # Hop length for feature extraction
MAX_MFCC_LENGTH = 200  # You can change this based on your dataset

# Mel Spectrogram Transformer
mel_transform = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)

def process_audio(file_path):
    try:
        # Load and preprocess
        waveform, sr = librosa.load(file_path, sr=16000, mono=True)
        if waveform.size == 0:
            print(f"Skipping empty audio: {file_path}")
            return None
        waveform, _ = librosa.effects.trim(waveform)
        if waveform.size == 0:
            print(f"Skipping silent audio: {file_path}")
            return None
        waveform = librosa.util.normalize(waveform)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)

        # Pad or truncate MFCCs to fixed length
        if mfcc.shape[1] < MAX_MFCC_LENGTH:
            # Pad with zeros
            pad_width = MAX_MFCC_LENGTH - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            mfcc = mfcc[:, :MAX_MFCC_LENGTH]

        return mfcc.astype(np.float32)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to process all files in a split using multiprocessing
def process_split(split):
    data = []
    split_path = os.path.join(dataset_root, split)

    for label, label_num in labels.items():
        label_path = os.path.join(split_path, label)
        files = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.endswith((".mp3", ".wav"))]

        # Parallel processing
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            results = list(tqdm(pool.imap(process_audio, files), total=len(files), desc=f"Processing {split}/{label}"))

        # Filter out None results
        for i, feature in enumerate(results):
            if feature is not None:
                data.append((feature, label_num))  # Append (feature, label) tuple
    
    features = [x[0] for x in data]
    label_array = np.array([x[1] for x in data]) #  creates a 1D NumPy array
    feature_array = np.stack(features) # Combines a list of 2D NumPy arraysinto a single 3D array.
    np.save(f"{split}_features.npy", feature_array, allow_pickle=True)
    np.save(f"{split}_labels.npy", label_array)

    print(f"Saved {split}: {len(features)} samples")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Ensures proper multiprocessing startup on Windows

    # Run preprocessing for all splits
    for split in splits:
        process_split(split)
    
    print("All preprocessing done!")
