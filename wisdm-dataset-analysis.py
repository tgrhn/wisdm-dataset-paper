# Tugrahan Karakadioglu, Ozyegin University 2024.
# This code is for the analysis of the dataset. 
# The shape of the dataset is: (1098208, 6) (userID, activity, time, acc_x_axis, acc_y_axis, acc_z_axis)
# For each activity, the activity counts are:
# Walking       424399
# Jogging       342179
# Upstairs      122869
# Downstairs    100427
# Sitting        59939
# Standing       48395
# Overall 1098208 activity.
# There are no null row in the dataset.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
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

def plot_activity_distribution(dataset):
    """Plot activity distribution as a bar chart."""
    activity_counts = dataset['activity'].value_counts()
    plt.figure(figsize=(10, 6))
    activity_counts.plot(kind='bar', color='skyblue')
    plt.title('Activity Distribution', fontsize=16)
    plt.xlabel('Activity', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.show()

def plot_accelerometer_data(dataset, activity, sample_size=200):
    """Plot accelerometer data for a specific activity."""
    activity_data = dataset[dataset['activity'] == activity].iloc[:sample_size]
    plt.figure(figsize=(12, 6))
    plt.plot(activity_data['time'], activity_data['acc_x_axis'], label='X-axis', alpha=0.8)
    plt.plot(activity_data['time'], activity_data['acc_y_axis'], label='Y-axis', alpha=0.8)
    plt.plot(activity_data['time'], activity_data['acc_z_axis'], label='Z-axis', alpha=0.8)
    plt.title(f'Accelerometer Data for {activity}', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Accelerometer Values', fontsize=12)
    plt.legend()
    plt.show()

def plot_correlation_heatmap(dataset):
    """Plot a heatmap for accelerometer axes correlations."""
    correlation_matrix = dataset[['acc_x_axis', 'acc_y_axis', 'acc_z_axis']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Accelerometer Axes', fontsize=16)
    plt.show()

def plot_user_activity_heatmap(dataset):
    """Plot a heatmap showing activity distribution across users."""
    user_activity_counts = dataset.groupby(['userID', 'activity']).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(user_activity_counts, cmap='YlGnBu', linewidths=0.5)
    plt.title('Activity Distribution Across Users', fontsize=16)
    plt.xlabel('Activity', fontsize=12)
    plt.ylabel('User ID', fontsize=12)
    plt.show()

def main():
    """Main function to execute the analysis."""
    dataset = load_data(file_path)
    dataset = preprocess_data(dataset)
    print(dataset.info())

    plot_activity_distribution(dataset)
    for activity in ['Walking', 'Jogging', 'Standing', 'Sitting', 'Downstairs', 'Upstairs']:
        plot_accelerometer_data(dataset, activity)
    plot_correlation_heatmap(dataset)
    plot_user_activity_heatmap(dataset)

if __name__ == "__main__":
    main()

