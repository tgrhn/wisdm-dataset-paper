# Tugrahan Karakadioglu, Ozyegin University 2024.
# In this code, we performed segment analysis for each
# 3-sec, 5-sec, and 10-sec. 
# Then split our dataset into train, test, and validation set.
# Also, we performed label encoding
# The new labels are:
# Class 0: Downstairs
# Class 1: Jogging
# Class 2: Sitting
# Class 3: Standing
# Class 4: Upstairs
# Class 5: Walking

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

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

def segment_data(data, window_size, overlap):
    """Segment data into fixed-size windows with overlap."""
    segments = []
    labels = []
    step = window_size - overlap

    for user in data['userID'].unique():
        user_data = data[data['userID'] == user]
        for i in range(0, len(user_data) - window_size, step):
            segment = user_data[['acc_x_axis', 'acc_y_axis', 'acc_z_axis']].iloc[i:i + window_size].values
            label = user_data['activity'].iloc[i + window_size - 1] 
            segments.append(segment)
            labels.append(label)
    if len(segments) == 0:
        raise ValueError("No segments were created. Check window size and overlap parameters.")
    return np.array(segments), np.array(labels)

def scale_data(X_train, X_val, X_test):
    """Flatten and scale data."""
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

dataset = load_data(file_path)
dataset = preprocess_data(dataset)

window_size = 60  # 10 seconds
overlap = 50  # 50% overlap
X, y = segment_data(dataset, window_size, overlap)
print(f"Shape of segmented data: {X.shape}, Labels: {len(y)}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Encoded Labels:", le.classes_) 
class_mapping = dict(zip(range(len(le.classes_)), le.classes_))
print("Class Mapping:")
for key, value in class_mapping.items():
    print(f"Class {key}: {value}")

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_data(X_train, X_val, X_test)

with open('preprocessed_wisdm_40_window_size.pkl', 'wb') as f:
    pickle.dump((X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test), f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('scaler_40.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Data preprocessing complete and saved!")

def print_class_distribution(y, set_name):
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution in {set_name}:")
    for label, count in zip(unique, counts):
        print(f"Class {label}: {count}")

print_class_distribution(y_train, "Training Set")
print_class_distribution(y_val, "Validation Set")
print_class_distribution(y_test, "Test Set")

def plot_segment(segment, title="Sample Segment"):
    plt.figure(figsize=(10, 6))
    plt.plot(segment[:, 0], label="X-axis")
    plt.plot(segment[:, 1], label="Y-axis")
    plt.plot(segment[:, 2], label="Z-axis")
    plt.title(title)
    plt.xlabel("Time (samples)")
    plt.ylabel("Accelerometer Values")
    plt.legend()
    plt.show()

plot_segment(X_train[0], title="Example Segment for Activity")
