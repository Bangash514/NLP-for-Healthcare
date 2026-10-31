# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:07:43 2024

@author: Administrator
"""

import pandas as pd
import chardet
import csv

# Specify the path to your file
file_path = r"C:\Users\Administrator\Downloads\NLPForHealthcare\DataPath\外科5-14000.csv"

# Step 1: Detect file encoding with chardet
with open(file_path, 'rb') as file:
    result = chardet.detect(file.read())
    encoding = result['encoding']
    print("Detected encoding:", encoding)

# Step 2: Read the file using csv library to handle commas within text
try:
    # Open the file with detected encoding
    with open(file_path, 'r', encoding=encoding, errors='replace') as file:
        # Use csv reader to read rows properly handling commas within text
        reader = csv.reader(file)
        rows = list(reader)
    
    # Convert to DataFrame
    data = pd.DataFrame(rows[1:], columns=rows[0])  # First row as column headers
    print("Data loaded into DataFrame successfully.")
    
except UnicodeDecodeError:
    print(f"Error reading file with detected encoding ({encoding}). Trying with GBK.")
    try:
        # Retry with GBK encoding
        with open(file_path, 'r', encoding='GBK', errors='replace') as file:
            reader = csv.reader(file)
            rows = list(reader)
        
        data = pd.DataFrame(rows[1:], columns=rows[0])
        print("Data loaded into DataFrame successfully with GBK encoding.")
    
    except Exception as e:
        print("Failed to load data with GBK encoding:", e)
        data = None

# Step 4: Display the data if successfully loaded
if data is not None:
    print("First few rows of the loaded data:\n", data.head())
else:
    print("Failed to load the data.")
