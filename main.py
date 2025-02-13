import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Flatten, GaussianNoise
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

# Function to extract MFCC features with time masking
def extract_features(file_path, max_pad_length=100, apply_augmentation=False):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    if mfccs.shape[1] < max_pad_length:
        pad_width = max_pad_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_length]
    
    if apply_augmentation:
        mask_start = np.random.randint(0, max_pad_length - 10)
        mask_length = np.random.randint(5, 15)
        mfccs[:, mask_start:mask_start + mask_length] = 0  # Apply time masking
    
    return mfccs.T  # Shape: (time_steps, features)

# Function to load dataset
def load_data(data_dir, augment=False):
    labels = []
    features = []
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    if not class_names:
        raise ValueError("No class directories found in the dataset!")

    class_mapping = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]

        if not files:
            print(f"Warning: No audio files found in {class_dir}")
            continue

        for file in files:
            file_path = os.path.join(class_dir, file)
            features.append(extract_features(file_path, apply_augmentation=False))  # Original data
            labels.append(class_mapping[class_name])

            if augment:
                features.append(extract_features(file_path, apply_augmentation=True))  # Augmented data
                labels.append(class_mapping[class_name])

    if not labels:
        raise ValueError("No valid audio files found!")

    return np.array(features), np.array(labels), class_mapping

# Set dataset path
data_dir = r"C:/Users/GANESH/OneDrive/Desktop/sao prj/data"

# Load dataset with augmentation5
X, y, class_mapping = load_data(data_dir, augment=True)

# Convert labels to one-hot encoding
y = to_categorical(y)   

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Save the scaler for later use
joblib.dump(scaler, "scaler.pkl")
joblib.dump(class_mapping, "class_mapping.pkl")

# Define the improved CNN + Bi-LSTM model
input_shape = (X.shape[1], X.shape[2])  # (time_steps, features)

model = Sequential([
    GaussianNoise(0.1, input_shape=input_shape),  # Noise for regularization

    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001))),
    Dropout(0.5),

    Bidirectional(LSTM(64, kernel_regularizer=l2(0.001))),
    Dropout(0.5),

    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),

    Dense(y.shape[1], activation='softmax')
])

# Compile model with AdamW optimizer
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler & early stopping
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

# Save trained model
model.save("VocalDisease_detection_model.h5")

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# ==========================
# PLOTTING CHARTS
# ==========================
def plot_training_history(history):
    epochs = range(len(history.history['accuracy']))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(epochs, history.history['val_accuracy'], label='Val Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history.history['val_loss'], label='Val Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(history)