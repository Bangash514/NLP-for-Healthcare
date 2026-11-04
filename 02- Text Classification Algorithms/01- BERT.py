# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:07:43 2024
@author: Administrator
"""
#Bangash PhD Scholar

#Text Classification by 'BERT'
#

import pandas as pd
import chardet
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import logging
import transformers  # Import the transformers library

# Set logging level to ERROR to suppress unnecessary warnings
logging.basicConfig(level=logging.ERROR)
transformers.logging.set_verbosity_error()

# Specify the new path to your file
file_path = r"E:\PhD 2022-26\西安交通大学\2024-25\NLPForHealthcare\DataPath\外科5-14000.csv"

# Step 1: Detect file encoding with chardet
with open(file_path, 'rb') as file:
    result = chardet.detect(file.read())
    encoding = result['encoding']
    print("Detected encoding:", encoding)

# Step 2: Read the file using csv library to handle commas within text
try:
    with open(file_path, 'r', encoding=encoding, errors='replace') as file:
        reader = csv.reader(file)
        rows = list(reader)
    data = pd.DataFrame(rows[1:], columns=rows[0])  # First row as column headers
    print("Data loaded into DataFrame successfully.")
    
except UnicodeDecodeError:
    print(f"Error reading file with detected encoding ({encoding}). Trying with GBK.")
    try:
        with open(file_path, 'r', encoding='GBK', errors='replace') as file:
            reader = csv.reader(file)
            rows = list(reader)
        data = pd.DataFrame(rows[1:], columns=rows[0])
        print("Data loaded into DataFrame successfully with GBK encoding.")
    except Exception as e:
        print("Failed to load data with GBK encoding:", e)
        data = None

if data is not None:
    print("First few rows of the loaded data:\n", data.head())
else:
    print("Failed to load the data.")

# BERT Training Code
if data is not None:
    # Select relevant columns
    X = data['ask'].tolist()  # Convert to list for BERT
    y = data['department'].astype('category').cat.codes  # Convert categories to numerical labels

    # Split the data without stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
    
    # Define the local path to the downloaded BERT model files
    local_model_path = r" "

    # Load BERT tokenizer and model from local files
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    model = BertForSequenceClassification.from_pretrained(local_model_path, num_labels=len(y.unique()))

    # Tokenize the inputs with reduced max_length
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=32)  # reduced max_length
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=32)    # reduced max_length

    # Create PyTorch datasets
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(train_encodings, y_train.tolist())
    test_dataset = Dataset(test_encodings, y_test.tolist())

    # Set up training arguments and Trainer API with 1 epoch for quick testing
    training_args = TrainingArguments(
        output_dir=r' ',  # New output directory
        num_train_epochs=1,              # number of training epochs set to 1
        per_device_train_batch_size=4,   # reduced batch size for training
        per_device_eval_batch_size=4,    # reduced batch size for evaluation
        warmup_steps=500,                # number of warmup steps
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        evaluation_strategy="epoch",      # evaluate each epoch
        save_strategy="no",               # Disable saving model checkpoints
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # Train the model with 1 epoch
    trainer.train()

    # Evaluate the model
    outputs = trainer.predict(test_dataset)
    y_pred = outputs.predictions.argmax(axis=1)

    # Print classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
