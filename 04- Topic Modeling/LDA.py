# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 05:40:39 2024
@author: Administrator
"""

import pandas as pd
import chardet
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging

# Set logging level to ERROR to suppress unnecessary warnings
logging.basicConfig(level=logging.ERROR)

# Specify the path to your file
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

# LDA Topic Modeling Code
if data is not None:
    # Select relevant columns for text analysis
    texts = data['ask'].tolist()  # Assume the 'ask' column contains the text data

    # Text Preprocessing and Vectorization
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)  # Adjust min/max df as needed
    X = vectorizer.fit_transform(texts)  # Create a document-term matrix

    # Train LDA model
    n_topics = 5  # Set the number of topics to discover
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    # Display Topics
    def display_topics(model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print(f"Topic {topic_idx + 1}:")
            print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

    no_top_words = 10  # Number of words to display per topic
    display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)
