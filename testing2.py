import os
import numpy as np
import librosa
import joblib  # To load scaler
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Function to extract MFCC features
def extract_features(file_path, max_pad_length=100):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Pad or truncate MFCCs to a fixed length
    if mfccs.shape[1] < max_pad_length:
        pad_width = max_pad_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_length]

    return mfccs.T  # Transpose to (time_steps, features)

# Function to predict disease for multiple files
def predict_stress_multiple(model_path, folder_path, scaler, class_mapping):
    """Load model and predict Diseases for multiple audio files in a folder."""
    model = load_model(model_path)
    class_labels = {v: k for k, v in class_mapping.items()}
    
    results = {}
    
    # Process each .wav file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path)
            features = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
            prediction = model.predict(features[np.newaxis, ...])
            predicted_class = np.argmax(prediction)
            results[file_name] = class_labels.get(predicted_class, "Unknown")
    
    return results

# ==================
# Load Trained Model
# ==================
model_path = "C:/Users/GANESH/OneDrive/Desktop/sao prj/VocalDisease_detection_model.h5"
model = load_model(model_path)

# ==============================
# Load Scaler and Class Mapping
# ==============================
scaler = joblib.load("scaler.pkl")  # Ensure this file was saved during training
class_mapping = joblib.load("class_mapping.pkl")  # Ensure this file was saved during training

# ==================
# Run Prediction
# ==================
folder_path = r"C:/Users/GANESH/OneDrive/Desktop/sao prj/data/Normal"  # Path to folder containing test .wav files
predictions = predict_stress_multiple(model_path, folder_path, scaler, class_mapping)

# Print results
for file, label in predictions.items():
    print(f"{file}: Predicted Disease -> {label}")