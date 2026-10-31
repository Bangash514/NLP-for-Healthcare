# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:07:37 2024

@author: Administrator
"""
#Bangash PhD Scholar
#CRF model for Named Entity Recognition (NER) on the MIMIC dataset using distant supervision with dictionaries
#Conditional Random Field (CRF) 

import os
import re
import pandas as pd
import sklearn_crfsuite
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Path to the folder with CSV files
data_path = r'C:\Users\Administrator\Downloads\NLPForHealthcare\mimic-iii'

# Load NOTEEVENTS.csv and display column names
data_file = os.path.join(data_path, 'NOTEEVENTS.csv')
df = pd.read_csv(data_file, low_memory=False)
print("Column names in NOTEEVENTS.csv:", df.columns)

# Check if 'text' column exists directly after loading
if 'text' not in df.columns:
    raise KeyError("The 'text' column is not present in NOTEEVENTS.csv. Please verify the dataset.")

# Strip whitespace and convert text to strings
df['text'] = df['text'].astype(str).str.strip()

# Filter out empty or short entries
df = df[df['text'].apply(lambda x: len(x) > 10)]

# Confirm 'text' column still exists after filtering
if 'text' not in df.columns:
    raise KeyError("The 'text' column is missing after filtering. Check the data processing steps.")

# Adjust sample size based on available data
if len(df) < 100:
    print(f"Warning: Only {len(df)} valid text entries available. Adjusting sample size.")
    sample_size = len(df)
else:
    sample_size = 100

# Define dictionaries for distant supervision
drug_dict = ['aspirin', 'ibuprofen', 'acetaminophen', 'morphine', 'xenon']  # Extend with more drug names
anatomy_dict = ['heart', 'brain', 'retina', 'lung', 'liver']  # Extend with more anatomy terms
condition_dict = ['pain', 'infection', 'fracture', 'tumor', 'disease']  # Extend with more conditions

# Regular expression patterns for entity detection
drug_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(drug) for drug in drug_dict) + r')\b', re.IGNORECASE)
anatomy_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(term) for term in anatomy_dict) + r')\b', re.IGNORECASE)
condition_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(cond) for cond in condition_dict) + r')\b', re.IGNORECASE)

# Function to apply distant supervision and label entities
def label_entities(text):
    words = text.split()
    labels = []
    for word in words:
        if drug_pattern.search(word):
            labels.append('Drug')
        elif anatomy_pattern.search(word):
            labels.append('Anatomy')
        elif condition_pattern.search(word):
            labels.append('Condition')
        else:
            labels.append('O')  # Outside any entity
    return list(zip(words, labels))

# Feature extraction function for CRF
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

# Prepare data for CRF
def prepare_data(sentences):
    X = [[word2features(sent, i) for i in range(len(sent))] for sent in sentences]
    y = [[label for _, label in sent] for sent in sentences]
    return X, y

# Apply labeling and prepare data
# Sample the non-empty text data
sentences = [label_entities(text) for text in df['text'].sample(sample_size)]
X, y = prepare_data(sentences)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Predict and evaluate
y_pred = crf.predict(X_test)
y_test_flat = [label for sent in y_test for label in sent]
y_pred_flat = [label for sent in y_pred for label in sent]

# Classification report
print("Classification Report:")
print(classification_report(y_test_flat, y_pred_flat, target_names=['Drug', 'Anatomy', 'Condition']))