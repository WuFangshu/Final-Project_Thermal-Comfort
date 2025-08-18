import pandas as pd
import numpy as np

# Load the dataset from CSV file
df = pd.read_csv("autotherm_sample.csv")  # Replace with your actual file path

def parse_sequence(seq_str):
    """
    Convert a tilde-separated string (e.g., "123.4~456.7") to a list of floats.
    Returns [0.0] if parsing fails.
    """
    try:
        return [float(x) for x in seq_str.split('~')]
    except:
        return [0.0]

def encode_sample(row):
    """
    Encode a single row of AutoTherm data into a fixed-length numerical feature vector.
    Combines numeric, categorical, and sequential features.
    """
    features = []

    # Numeric features (single-value per column)
    numeric_columns = [
        'Age', 'Weight', 'Height', 'Bodyfat', 'Bodytemp', 
        'Sport-Last-Hour', 'Time-Since-Meal', 'Tiredness', 
        'Clothing-Level', 'Metabolic-Rate', 'Heart_Rate',
        'Ambient_Temperature', 'Ambient_Humidity', 'Solar_Radiation'
    ]
    for col in numeric_columns:
        features.append(float(row[col]))

    # Categorical feature: Gender (1 for male, 0 for female)
    gender = 1.0 if row['Gender'].strip().lower() == 'male' else 0.0
    features.append(gender)

    # Categorical features: Emotions (simple integer encoding)
    emotion_map = {'Neutral': 0, 'Fear': 1, 'Happy': 2, 'Sad': 3}
    for col in ['Emotion-Self', 'Emotion-ML']:
        emotion = row[col].split(',')[0].strip()
        features.append(emotion_map.get(emotion, -1))

    # Sequential features: Use mean and standard deviation of values
    sequence_columns = [
        'Radiation-Temp', 'PCE-Ambient-Temp', 'Air-Velocity',
        'Nose', 'Neck', 'RShoulder', 'RElbow', 'LShoulder', 'LElbow',
        'REye', 'LEye', 'REar', 'LEar', 'Wrist_Skin_Temperature', 'GSR'
    ]
    for col in sequence_columns:
        values = parse_sequence(row[col])
        features.append(np.mean(values))
        features.append(np.std(values))

    return np.array(features)
