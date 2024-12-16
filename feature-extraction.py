# Tugrahan Karakadioglu, Ozyegin University 2024.
# In this code, we performed segment analysis for each
# 3-sec, 5-sec, and 10-sec time windows.
# We extracted statistical features such as mean, std, min, and max
# and split the dataset into train, test, and validation sets.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

file_path = '/Users/tugrahankarakadioglu/Desktop/wisdm-dataset-paper/dataset/WISDM-dataset.txt'

def load_data(file_path):
    """Load and process dataset from the given file path."""
    with open(file_path, 'r') as file:
        rows = file.readlines()

    data_rows = []
    for i, line in enumerate(rows):
        try:
            line = line.split(',')
            last = line[5].split(';')[0].strip()
            if last == '':
                continue
            data_rows.append([line[0], line[1], line[2], line[3], line[4], last])
        except Exception as e:
            print(f"Error at line {i}: {e}")

    columns = ['userID', 'activity', 'time', 'acc_x_axis', 'acc_y_axis', 'acc_z_axis']
    dataset = pd.DataFrame(data=data_rows, columns=columns)
    return dataset

def preprocess_data(dataset):
    """Preprocess the dataset for analysis."""
    dataset['userID'] = dataset['userID'].astype(int)
    dataset['time'] = dataset['time'].astype(float)
    dataset[['acc_x_axis', 'acc_y_axis', 'acc_z_axis']] = dataset[['acc_x_axis', 'acc_y_axis', 'acc_z_axis']].astype(float)
    return dataset

from scipy.stats import skew, kurtosis

def segment_and_extract_features(data, window_size, overlap):
    """Segment data into fixed-size windows and extract statistical features."""
    step = window_size - overlap
    features = []
    labels = []

    for user in data['userID'].unique():
        user_data = data[data['userID'] == user]
        for i in range(0, len(user_data) - window_size, step):
            window = user_data[['acc_x_axis', 'acc_y_axis', 'acc_z_axis']].iloc[i:i + window_size]
            label = user_data['activity'].iloc[i + window_size - 1]

            # Extract statistical features for each axis
            feature_vector = []
            for axis in ['acc_x_axis', 'acc_y_axis', 'acc_z_axis']:
                feature_vector += [
                    window[axis].mean(),
                    window[axis].std(),
                    window[axis].min(),
                    window[axis].max(),
                    window[axis].median(),
                    skew(window[axis]),
                    kurtosis(window[axis]),
                    np.sqrt(np.mean(window[axis]**2)),  # RMS
                    np.sum(np.diff(np.sign(window[axis])) != 0)  # Zero-Crossing Rate
                ]
            features.append(feature_vector)
            labels.append(label)

    return np.array(features), np.array(labels)

if __name__ == "__main__":
    dataset = load_data(file_path)
    dataset = preprocess_data(dataset)

    window_size = 60  # 10 seconds (assuming 20 Hz sampling rate)
    overlap = 50       # 50% overlap

    X, y = segment_and_extract_features(dataset, window_size, overlap)
    print(f"Shape of extracted features: {X.shape}, Labels: {len(y)}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Encoded Labels:", le.classes_)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open('rf-preprocessed_wisdm_features_60_window_size.pkl', 'wb') as f:
        pickle.dump((X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test), f)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    with open('rf-scaler_60.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Data preprocessing and feature extraction complete!")

    def print_class_distribution(y, set_name):
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass distribution in {set_name}:")
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count}")

    print_class_distribution(y_train, "Training Set")
    print_class_distribution(y_val, "Validation Set")
    print_class_distribution(y_test, "Test Set")

    def plot_features(features, title="Sample Features"):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(features)), features)
        plt.title(title)
        plt.xlabel("Feature Index")
        plt.ylabel("Feature Value")
        plt.show()

    plot_features(X_train_scaled[0], title="Sample Features for a Window")
