# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:13:51 2024

#Bangash PhD Scholar
#Xi'an Jiaotong University


import os
import pandas as pd
import re

# Path to the folder with CSV files
data_path = r'C:\Users\Administrator\Downloads\NLPForHealthcare\mimic-iii'

# Load all CSV files into a dictionary of DataFrames
data_files = {}
for filename in os.listdir(data_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_path, filename)
        data_files[filename] = pd.read_csv(file_path)
        print(f"Loaded {filename} with shape: {data_files[filename].shape}")

# Enhanced dictionary of medical terms for dictionary-based entity recognition
medical_terms = {
    'Drug': ['aspirin', 'ibuprofen', 'paracetamol', 'xenon', 'laser', 'implant', 'morphine', 'warfarin'],
    'Condition': ['retinal', 'lesion', 'chorioretinal', 'diabetes', 'hypertension', 'asthma', 'infection'],
    'Procedure': ['destruction', 'photocoagulation', 'implant', 'biopsy', 'MRI', 'surgery', 'resection', 'excision']
}

# Define the rule-based entity extraction function with enhanced regular expressions
def rule_based_entity_extraction(text):
    entities = []
    # Regular expression for identifying dates in the format YYYY-MM-DD or numeric patterns
    date_pattern = r'\b\d{4}-\d{2}-\d{2}\b'
    if re.search(date_pattern, text):
        entities.append("Date")
    
    # Dictionary-based entity extraction
    for entity_type, terms in medical_terms.items():
        for term in terms:
            # Use case-insensitive search
            if term.lower() in text.lower():
                entities.append(f"{entity_type}: {term}")
    
    return entities

# Apply the entity extraction function to the 'D_ICD_PROCEDURES.csv' file
if 'D_ICD_PROCEDURES.csv' in data_files:
    procedures_df = data_files['D_ICD_PROCEDURES.csv']
    
    # Check for the column 'long_title' (or adjust if the column name is different)
    if 'long_title' in procedures_df.columns:
        # Apply entity extraction on each entry in the 'long_title' column
        procedures_df['Extracted Entities'] = procedures_df['long_title'].apply(rule_based_entity_extraction)
        
        # Display the extracted entities for each procedure
        print(procedures_df[['long_title', 'Extracted Entities']].head())
    else:
        print("Column 'long_title' not found in D_ICD_PROCEDURES.csv")
else:
print("D_ICD_PROCEDURES.csv file not found in the directory")
