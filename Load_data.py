import os
import pandas as pd

# Define dataset root directory
dataset_root = r"D:\OneDrive\Desktop\Projects\DeepFake\Speech Dataset\for-norm\for-norm"

# Define subdirectories
splits = ["Training", "Testing", "Validation"]
labels = {"fake": "spoof", "real": "bonafide"}  # Standardize labels for AASIST

# Function to clean file names (Removes any unwanted extra extensions.)
def clean_filename(filename):
    """Removes all extra extensions except .mp3 or .wav"""
    base_name, ext = os.path.splitext(filename)  # Split name and extension
    
    # Keep only .mp3 or .wav
    if ext in [".mp3", ".wav"]:
        return base_name + ext  
    else:
        return filename  

# Loop through splits and create CSV files
for split in splits:
    data = []
    for label in labels.keys():
        folder_path = os.path.join(dataset_root, split, label)
        
        # Check if directory exists
        if not os.path.exists(folder_path):
            print(f"Directory not found: {folder_path}")
            continue
        
        print(f"Scanning {folder_path}...")  # Debug
        
        for file in os.listdir(folder_path):
            if file.endswith((".mp3", ".wav")):  # Process both mp3 and wav files
                full_file_path = os.path.join(folder_path, file)  # Full path
                
                # Verify file exists before adding to CSV
                if os.path.exists(full_file_path):
                    cleaned_name = clean_filename(file)  # Clean filename
                    file_path = os.path.join(split, label, cleaned_name)  # Relative path
                    data.append([file_path, labels[label]])  # Append to list
                else:
                    print(f"Missing File: {full_file_path}") 

    print(f"Collected {len(data)} files for {split} set.")  

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(data, columns=["filename", "label"])
    df.to_csv(f"{split}.csv", index=False)

print("CSV files created successfully!")
