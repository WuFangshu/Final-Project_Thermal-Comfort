# step1_load_autotherm_full_with_progress.py
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os

# Print current working directory (to show where the CSV will be saved)
print("Current working directory:", os.getcwd())

# Load dataset from Hugging Face
print("Loading AutoTherm dataset from Hugging Face...")
dataset = load_dataset("kopetri/AutoTherm")
train_data = dataset["train"]

# Show number of records
print(f"Dataset loaded successfully with {len(train_data)} records.")

# Convert to pandas DataFrame
print("Converting dataset to pandas DataFrame...")
df = pd.DataFrame(train_data)

# Select relevant feature columns and label
features = [
    "Age", "Gender", "Weight", "Height", "Bodyfat", "Bodytemp",
    "Sport-Last-Hour", "Time-Since-Meal", "Tiredness",
    "Clothing-Level", "Radiation-Temp", "PCE-Ambient-Temp",
    "Air-Velocity", "Metabolic-Rate", "Emotion-Self", "Emotion-ML",
    "Nose", "Neck", "RShoulder", "RElbow", "LShoulder", "LElbow",
    "REye", "LEye", "REar", "LEar",
    "Wrist_Skin_Temperature", "Heart_Rate", "GSR",
    "Ambient_Temperature", "Ambient_Humidity", "Solar_Radiation"
]
label = "Label"

print("Filtering selected feature columns and label...")
df = df[features + [label]]

# Save as CSV
save_path = "autotherm_full_features.csv"
df.to_csv(save_path, index=False)

print(f"Saved to file: {save_path}")
print(f"Saved data shape: {df.shape[0]} rows, {df.shape[1]} columns")
