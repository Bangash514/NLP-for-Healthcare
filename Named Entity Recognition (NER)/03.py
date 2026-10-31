# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:53:05 2024

@author: Administrator
"""
import os
import re
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Path to the folder with CSV files
data_path = r'C:\Users\Administrator\Downloads\NLPForHealthcare\mimic-iii'

# Load all CSV files from the data path into a dictionary
data_files = {}
for filename in os.listdir(data_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_path, filename)
        try:
            df = pd.read_csv(file_path, low_memory=False)
            data_files[filename] = df
            print(f"Loaded {filename} with shape: {df.shape}")
        except Exception as e:
            print(f"Could not load {filename}: {e}")

# Sample labeled data for demonstration purposes (replace with actual labeled data from your dataset)
data = [
    [('Patient', 'O'), ('was', 'O'), ('given', 'O'), ('aspirin', 'Drug'), ('for', 'O'), ('pain', 'Condition')],
    [('MRI', 'Procedure'), ('revealed', 'O'), ('brain', 'Anatomy'), ('tumor', 'Condition')],
    # Add more sentences here or use actual labeled data from your dataset
]

# Split data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Feature extraction function
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'word.lower()': word.lower(),
        'is_upper': word.isupper(),
        'is_title': word.istitle(),
        'is_digit': word.isdigit(),
    }
    if i > 0:
        features.update({
            '-1:word.lower()': sent[i-1][0].lower(),
            '-1:is_upper': sent[i-1][0].isupper(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence
    if i < len(sent) - 1:
        features.update({
            '+1:word.lower()': sent[i+1][0].lower(),
            '+1:is_upper': sent[i+1][0].isupper(),
        })
    else:
        features['EOS'] = True  # End of sentence
    return features

# Prepare the training and test data for CRF
def prepare_data(data):
    X = [[word2features(sent, i) for i in range(len(sent))] for sent in data]
    y = [[label for _, label in sent] for sent in data]
    return X, y

X_train, y_train = prepare_data(train_data)
X_test, y_test = prepare_data(test_data)

# Train a CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,  # Coefficient for L1 regularization
    c2=0.1,  # Coefficient for L2 regularization
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Predict on the test set
y_pred = crf.predict(X_test)

# Flatten the lists for y_test and y_pred
y_test_flat = [label for sent in y_test for label in sent]
y_pred_flat = [label for sent in y_pred for label in sent]

# Generate and print the classification report
print("Classification Report:")
print(classification_report(y_test_flat, y_pred_flat, target_names=['Drug', 'Anatomy', 'Condition', 'Procedure'], zero_division=0))