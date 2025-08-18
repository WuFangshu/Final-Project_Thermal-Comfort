import pandas as pd
import numpy as np
from tqdm import tqdm
from tsaug import AddNoise, TimeWarp, Drift
import sys

# Configuration
csv_path = "autotherm_full_features.csv"
window_size = 30
stride = 10
output_aug = "autotherm_synthetic_all_augmented.npz"
batch_size = 50000  # number of sequences per augmentation batch

# Load CSV
print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path, low_memory=False)
print(f"Loaded successfully with {df.shape[0]} rows.")

# Split multi-value string columns (x~y~z)
multi_value_cols = ["Nose", "Neck", "RShoulder", "RElbow", "LShoulder", "LElbow", "REye", "LEye", "REar", "LEar"]
for col in multi_value_cols:
    if col in df.columns:
        print(f"Splitting column: {col}")
        parts = df[col].astype(str).str.split("~", expand=True)
        df[f"{col}_x"] = parts[0].astype(float)
        df[f"{col}_y"] = parts[1].astype(float)
        df[f"{col}_z"] = parts[2].astype(float)
        df.drop(columns=[col], inplace=True)

# Encode categorical features
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
emotion_map = {
    "Neutral": 0, "Happy": 1, "Sad": 2, "Angry": 3,
    "Surprised": 4, "Fearful": 5, "Disgusted": 6, "Contempt": 7
}
for emo_col in ["Emotion-Self", "Emotion-ML"]:
    if emo_col in df.columns:
        df[emo_col] = df[emo_col].map(emotion_map)

# Separate features and labels
label_col = "Label"
features = [c for c in df.columns if c != label_col]
X_all = df[features].astype(np.float32).values
y_all = df[label_col].astype(np.int64).values

# Sliding window segmentation
print("Performing sliding window segmentation...")
X_seq, y_seq = [], []
for i in tqdm(range(0, len(X_all) - window_size, stride)):
    X_seq.append(X_all[i:i+window_size])
    y_seq.append(y_all[i + window_size // 2])
X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.int64)
print(f"Windowing completed: {X_seq.shape}")

# Data augmentation using tsaug (in batches)
print(f"Starting data augmentation in batches of {batch_size} samples...")
augmenter = (
    AddNoise(scale=0.05) +
    TimeWarp(n_speed_change=3, max_speed_ratio=2) +
    Drift(max_drift=0.1)
)

X_aug_parts, y_aug_parts = [], []

for i in range(0, len(X_seq), batch_size):
    batch_X = X_seq[i:i+batch_size]
    batch_y = y_seq[i:i+batch_size]
    print(f"Augmenting batch {i // batch_size + 1}: {len(batch_X)} samples")
    try:
        batch_X_aug = augmenter.augment(batch_X).astype(np.float32)
        X_aug_parts.append(batch_X_aug)
        y_aug_parts.append(batch_y)
    except Exception as e:
        print(f"Batch {i} augmentation failed: {e}")

# Concatenate all augmented data
X_aug = np.concatenate(X_aug_parts, axis=0)
y_aug = np.concatenate(y_aug_parts, axis=0)
print(f"Augmentation completed. Total samples: {X_aug.shape}")

# Save to .npz file
np.savez(output_aug, X=X_aug, y=y_aug)
print(f"Saved to file: {output_aug}")
