import pickle
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import Counter

with open('/Users/tugrahankarakadioglu/Desktop/wisdm-dataset-paper/pickles/preprocessed_wisdm_100_window_size.pkl', 'rb') as f:
    X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

with open('/Users/tugrahankarakadioglu/Desktop/wisdm-dataset-paper/pickles/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

with open('/Users/tugrahankarakadioglu/Desktop/wisdm-dataset-paper/pickles/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

def scale_data(scaler, X_train, X_val, X_test):
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

X_train_scaled, X_val_scaled, X_test_scaled = scale_data(scaler, X_train, X_val, X_test)
print("Original Class Distribution:", Counter(y_train))

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print("Balanced Class Distribution:", Counter(y_train_balanced))

window_size = 100
num_features = 3
X_train_dl = X_train_balanced.reshape(-1, window_size, num_features)
X_val_dl = X_val_scaled.reshape(-1, window_size, num_features)
X_test_dl = X_test_scaled.reshape(-1, window_size, num_features)

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_dl.shape[1], X_train_dl.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y_train)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train_dl, y_train_balanced, validation_data=(X_val_dl, y_val), epochs=20, batch_size=64)

test_loss, test_acc = model.evaluate(X_test_dl, y_test)
print(f"\nLSTM-CNN Test Accuracy After SMOTE: {test_acc:.2f}")

y_test_pred = np.argmax(model.predict(X_test_dl), axis=1)

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix for LSTM-CNN After SMOTE")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("LSTM-CNN Training and Validation Accuracy After SMOTE")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
