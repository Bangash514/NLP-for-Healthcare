# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 22:43:41 2024

@author: Administrator
"""


#Hierarchical Dirichlet Process (HDP)
import pandas as pd
import chardet
import csv
from gensim import corpora, models
from gensim.models.hdpmodel import HdpModel
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Set logging level to ERROR to suppress unnecessary warnings
logging.basicConfig(level=logging.ERROR)

# Specify the path to your file
file_path = r" "

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

# HDP Topic Modeling Code
if data is not None:
    # Select relevant columns for text analysis
    texts = data['ask'].tolist()  # Assume the 'ask' column contains the text data

    # Preprocess text data for HDP
    processed_texts = [text.split() for text in texts]  # Simple tokenization by splitting
    
    # Create a dictionary and corpus for HDP
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Train HDP model
    hdp_model = HdpModel(corpus=corpus, id2word=dictionary)

    # Display Topics
    num_topics = 5  # Number of topics to display
    topics = hdp_model.print_topics(num_topics=num_topics, num_words=10)
    
    for i, topic in topics:
        print(f"Topic {i + 1}: {topic}")
