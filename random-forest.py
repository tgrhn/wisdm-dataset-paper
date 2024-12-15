import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

with open('/Users/tugrahankarakadioglu/Desktop/wisdm-dataset-paper/pickles/preprocessed_wisdm_100_window_size.pkl', 'rb') as f:
    X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

with open('/Users/tugrahankarakadioglu/Desktop/wisdm-dataset-paper/pickles/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

with open('/Users/tugrahankarakadioglu/Desktop/wisdm-dataset-paper/pickles/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

def scale_data_with_scaler(scaler, X_train, X_val, X_test):
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

X_train_scaled, X_val_scaled, X_test_scaled = scale_data_with_scaler(scaler, X_train, X_val, X_test)

print("Original class distribution:", Counter(y_train))

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("Balanced class distribution:", Counter(y_train_balanced))

rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=20, 
    min_samples_split=10, 
    min_samples_leaf=5,  
    max_features='sqrt', 
    random_state=42
)
rf_model.fit(X_train_balanced, y_train_balanced)

y_val_pred = rf_model.predict(X_val_scaled)
y_test_pred = rf_model.predict(X_test_scaled)

val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("Random Forest Results After SMOTE:")
print(f"Validation Accuracy: {val_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix for Random Forest After SMOTE")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
