import numpy as np 
import librosa
import joblib  # To save and load scaler
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

# Function to predict Disease
def predict_stress(model_path, test_file, scaler, class_mapping):
    """Load model and predict Disease from an audio file."""
    model = load_model(model_path)

    # Extract features from test audio file
    features = extract_features(test_file)
    features = features[np.newaxis, ..., np.newaxis]  # Reshape to match model input shape

    # Normalize using the loaded scaler
    features = scaler.transform(features.reshape(-1, features.shape[-2])).reshape(features.shape)

    # Make prediction
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)

    # Get class label
    class_labels = {v: k for k, v in class_mapping.items()}
    return class_labels.get(predicted_class, "Unknown")

# ==================
# Load Trained Model
# ================== 
model = load_model("C:/Users/GANESH/OneDrive/Desktop/sao prj/VocalDisease_detection_model.h5")

# ==============================
# Load Scaler and Class Mapping
# ==============================
# Load pre-trained scaler
scaler = joblib.load("scaler.pkl")  # Ensure this file was saved during training

# Load class mapping
class_mapping = joblib.load("class_mapping.pkl")  # Ensure this file was saved during training

# ==================
# Run Prediction
# ==================
# Define test audio file path
test_audio_path = r"C:/Users/GANESH/OneDrive/Desktop/sao prj/data/Laryngozele/1449-a_lhl.wav"

# Predict Disease
predicted_label = predict_stress("C:/Users/GANESH/OneDrive/Desktop/sao prj/VocalDisease_detection_model.h5", 
                                 test_audio_path, scaler, class_mapping)
print(f"Predicted Disease: {predicted_label}")
