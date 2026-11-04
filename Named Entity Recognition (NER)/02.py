# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:35:32 2024

@author: Administrator
"""
#Rule-Based and Dictionary-Based Named Entity Recognition for Healthcare Data

import os
import re
import pandas as pd

# Path to the folder with CSV files
data_path = r'C:\Users\mimic-iii'

# Define dictionaries and patterns for entity recognition
drug_dict = ['aspirin', 'ibuprofen', 'acetaminophen', 'morphine', 'xenon']  # Add known drug names
anatomical_dict = ['heart', 'brain', 'retina', 'lung', 'liver']  # Add known anatomical terms

# Regular expression patterns for entities
drug_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(drug) for drug in drug_dict) + r')\b', re.IGNORECASE)
anatomical_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(term) for term in anatomical_dict) + r')\b', re.IGNORECASE)

# Function to apply rule-based NER
def extract_entities(text):
    drugs = drug_pattern.findall(text)
    anatomical_terms = anatomical_pattern.findall(text)
    entities = {'Drug': list(set(drugs)), 'Anatomy': list(set(anatomical_terms))}
    return entities

# Load CSV files and apply entity recognition
data_files = {}
for filename in os.listdir(data_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_path, filename)
        try:
            df = pd.read_csv(file_path, low_memory=False)
            data_files[filename] = df
            print(f"Loaded {filename} with shape: {df.shape}")
            
            # Check if a column with textual data exists (e.g., descriptions)
            if 'long_title' in df.columns:
                df['Extracted Entities'] = df['long_title'].fillna('').apply(extract_entities)
                print(df[['long_title', 'Extracted Entities']].head())
        except Exception as e:
            print(f"Could not load {filename}: {e}")

# Display sample results from a specific table (e.g., D_ICD_PROCEDURES)
if 'D_ICD_PROCEDURES.csv' in data_files:
    df_procedures = data_files['D_ICD_PROCEDURES.csv']
    print("Sample from D_ICD_PROCEDURES with extracted entities:")
    print(df_procedures[['long_title', 'Extracted Entities']].head())
