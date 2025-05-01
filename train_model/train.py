import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Activation, Conv1D, BatchNormalization, MaxPooling1D, Flatten
from scipy.signal import find_peaks
import sys

import pyaudio
from collections import deque

NMFCC=13

import os
import librosa
import numpy as np
import time

# Constants
HOP_LENGTH = 20
SAMPLE_RATE = 8000  # 8 kHz for mono audio
CHANNELS = 1        # Mono audio
FORMAT = pyaudio.paFloat32  # Float32 audio format
CHUNK_SIZE = 2048   # Size of each audio chunk
MAX_UTTERANCE_TIME = 2  # Max duration of utterance in seconds for buffering
MAX_FRAMES = 600     # Max number of MFCC frames (as per previous model setup)

def normalize(a):
	max_peak = np.max(np.abs(a))
	if max_peak > 0.0001:
		ratio = 1 / max_peak
		return a * ratio
	return a

def estimate_formants_from_file(audio_path):
    return np.array([0., 0., 0.])
    y, sr = librosa.load(audio_path, sr=None)
    y = normalize(y)
    
    formant_sequence = []

    for start in range(0, len(y) - CHUNK_SIZE + 1, HOP_LENGTH):
        chunk = y[start:start + CHUNK_SIZE]

        # Skip silent frames
        if np.max(np.abs(chunk)) < 1e-4:
            continue

        D = np.abs(librosa.stft(chunk, n_fft=CHUNK_SIZE, center=False))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=CHUNK_SIZE)

        # Get spectral peaks
        spectrum = D.mean(axis=1)
        peaks, _ = find_peaks(spectrum, distance=20)

        # Extract first 3 formant frequencies
        if len(peaks) >= 3:
            formants = freqs[peaks[:3]]
        else:
            formants = np.pad(freqs[peaks], (0, 3 - len(peaks)), mode='constant')

        formant_sequence.append(formants)

    formant_sequence = np.array(formant_sequence)
    num_frames = formant_sequence.shape[0]
   
    print(f" =============== {num_frames}") 
    if num_frames < MAX_FRAMES:
        padding = MAX_FRAMES - num_frames
        formant_sequence = np.pad(formant_sequence, ((0, padding), (0, 0)), mode='constant') 
    elif num_frames > MAX_FRAMES:
        print(f" =============== ERROR =============== {num_frames}")
        formant_sequence = formant_sequence[:MAX_FRAMES, :]
        
    return formant_sequence.astype(np.float32)


# A simple function to convert audio to MFCC
def audio_buffer_to_mfcc(audio_data, n_mfcc=NMFCC, hop_length=HOP_LENGTH, n_fft=512, max_frames=MAX_FRAMES):
    # Assuming audio_data is a numpy array of shape (num_samples,)

    audio_data = normalize(audio_data)

    mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

    mfcc = mfcc.T  # Each row is a frame with NMFCC MFCC features
    
    num_frames = mfcc.shape[0]
    
    if num_frames < max_frames:
        padding = max_frames - num_frames
        mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant')
    elif num_frames > max_frames:
        mfcc = mfcc[:max_frames, :]
    
    return mfcc.astype(np.float32)

def audio_to_mfcc(audio_file, n_mfcc=NMFCC, hop_length=HOP_LENGTH, n_fft=512, max_frames=MAX_FRAMES):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)  # sr=None ensures original sample rate

    y = normalize(y)   
 
    # Compute MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    
    # Transpose MFCC so that the shape is (frames, n_mfcc)
    mfcc = mfcc.T  # Each row is a frame with NMFCC features
    
    # Ensure the output has a fixed number of frames (max_frames)
    num_frames = mfcc.shape[0]

    print(f" =============== samples: {len(y)}  frames: {num_frames} {audio_file}")
    if num_frames < max_frames:
        # Pad with zeros if there are fewer frames than max_frames
        padding = max_frames - num_frames
        mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant')
    elif num_frames > max_frames:
        # Truncate to max_frames if there are more frames than max_frames
        mfcc = mfcc[:max_frames, :]
    
    mfcc = mfcc.astype(np.float32)
    
    return mfcc

# Function to load audio data from directory, convert to MFCCs, and assign labels
def load_dataset(base_dir, n_mfcc=NMFCC, hop_length=HOP_LENGTH, n_fft=512, max_frames=MAX_FRAMES):
    # Initialize variables to store data
    category_dict = {}
    train_mfcc_data = []
    train_formants_data = []
    train_labels = []
    test_mfcc_data = []
    test_formants_data = []
    test_labels = []

    # Iterate through categories in the "Train" and "Test" subdirectories
    for label_idx, category in enumerate(os.listdir(os.path.join(base_dir, 'Train'))):
        category_path_train = os.path.join(base_dir, 'Train', category)
        print(f" -- label_idx: {label_idx}, category: {category}, category_path_train: {category_path_train} ")
        category_path_test = os.path.join(base_dir, 'Test', category)
        print(f" -- label_idx: {label_idx}, category: {category}, category_path_test: {category_path_test} ")

        # Add category to category_dict
        category_dict[label_idx] = category

        # Process training files in this category
        for audio_file in os.listdir(category_path_train):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(category_path_train, audio_file)
                mfcc = audio_to_mfcc(audio_path, n_mfcc, hop_length, n_fft, max_frames)
                formants = estimate_formants_from_file(audio_path)
                train_mfcc_data.append(mfcc)
                train_formants_data.append(formants)
                train_labels.append(label_idx)

        # Process testing files in this category
        for audio_file in os.listdir(category_path_test):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(category_path_test, audio_file)
                mfcc = audio_to_mfcc(audio_path, n_mfcc, hop_length, n_fft, max_frames)
                formants = estimate_formants_from_file(audio_path)
                test_mfcc_data.append(mfcc)
                test_formants_data.append(formants)
                test_labels.append(label_idx)

    # Convert data lists to numpy arrays
    train_mfcc_data = np.array(train_mfcc_data)
    train_labels = np.array(train_labels)
    train_formants_data = np.array(train_formants_data)
    test_mfcc_data = np.array(test_mfcc_data)
    test_labels = np.array(test_labels)
    test_formants_data = np.array(test_formants_data)

    return category_dict, train_mfcc_data, train_formants_data, train_labels, test_mfcc_data, test_formants_data, test_labels


# Example usage:
base_directory = './DatasetMonosyllables'
category_dict, train_mfcc_data, train_formants_data, train_labels, test_mfcc_data, test_formants_data, test_labels = load_dataset(base_directory)

# Print results (example)
print("Category Dictionary:", category_dict)
print("MFCC Data Shape:", train_mfcc_data.shape)
print("Formants Data Shape:", train_formants_data.shape)
print("Labels Shape:", train_labels.shape)
print("Labels: ", train_labels)

print("Test MFCC Data Shape:", test_mfcc_data.shape)
print("Test Labels Shape:", test_labels.shape)
print("Test Labels: ", test_labels)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, MaxPooling1D, Flatten,
                                     Dense, Dropout, Concatenate)

from tensorflow.keras.optimizers import Adam

# Initialize Adam with a custom learning rate
adam_optimizer = Adam(learning_rate=0.003)

def cnn1d_model(input_shape=(100, 13), num_classes=10):
    model = Sequential([
        # 1st Conv Layer
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),

        # 2nd Conv Layer
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),

        # 3rd Conv Layer
        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),

	# 3rd Conv Layer
        Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),

        # Flatten for Fully Connected Layers
        Flatten(),

        # Fully Connected Layers # 128 and halves
        Dense(512, activation='relu'),
        Dropout(0.25),

        Dense(256, activation='relu'),
        Dropout(0.25),

	Dense(64, activation='relu'),
        Dropout(0.25),

	Dense(16, activation='relu'),
        Dropout(0.25),

        # Output Layer
        Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')  # Multi-class or binary
    ])

    # Compile Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy',
                  metrics=['accuracy'])

    return model

model = cnn1d_model((MAX_FRAMES, NMFCC), num_classes=3)

train_mfcc_data = np.array(train_mfcc_data)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
train_formants_data = np.array(train_formants_data)
test_formants_data = np.array(test_formants_data)
print(train_mfcc_data.shape)
print(train_mfcc_data[0][0].dtype)
print(train_mfcc_data.dtype)
print(test_mfcc_data.shape)
print(test_mfcc_data[0][0].dtype)
print(test_mfcc_data.dtype)
print(train_labels)
print(test_labels)
import sys

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation loss
    patience=45,          # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore model weights from the best epoch
)

model.fit(train_mfcc_data, train_labels, batch_size=32, epochs=240, validation_data=(test_mfcc_data, test_labels), callbacks=[early_stopping] )
print(test_labels)
print(category_dict)
print("preds: ")
print(model.predict(test_mfcc_data[0:5]))
print(np.argmax(model.predict(test_mfcc_data), axis=1))
print(" ")
print(test_labels)

model.save('monosyllables_model_v3.keras')

