# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 03:36:26 2024

@author: Administrator
"""
# SVM and enhanced TF-IDF settings:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# Specify the path to your CSV file
file_path = r"E:\PhD 2022-26\西安交通大学\2024-25\NLPForHealthcare\DataPath\外科5-14000.csv"

# Load the CSV file
try:
    df = pd.read_csv(file_path, encoding='gb18030')
    print("CSV file loaded successfully.")
    
    # Assuming 'answer' is the text column and 'department' is the label column
    X = df['answer']  # Features (text data)
    y = df['department']  # Labels (classification target)
    
    # Combine features and labels into a single DataFrame for resampling
    df_balanced = df[['answer', 'department']].copy()

    # Identify the largest and smallest classes for resampling
    largest_class = df_balanced['department'].value_counts().idxmax()
    smallest_class = df_balanced['department'].value_counts().idxmin()
    
    # Separate the majority and minority classes
    df_majority = df_balanced[df_balanced['department'] == largest_class]
    df_minority = df_balanced[df_balanced['department'] == smallest_class]

    # Downsample majority class and upsample minority class
    df_majority_downsampled = resample(df_majority, 
                                       replace=False,
                                       n_samples=len(df_minority) * 10, 
                                       random_state=42)
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,
                                     n_samples=50, 
                                     random_state=42)

    # Combine the balanced classes with the original data
    df_balanced = pd.concat([df_balanced, df_majority_downsampled, df_minority_upsampled])

    # Split the balanced dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df_balanced['answer'], df_balanced['department'], test_size=0.2, random_state=42)

    # Vectorize the text data using TF-IDF with bigrams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)  # Adjust max_features as needed
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Initialize and train Naive Bayes classifier
    nb_classifier = MultinomialNB(alpha=0.5)  # Adjust alpha for tuning
    nb_classifier.fit(X_train_tfidf, y_train)

    # Predict on the test data with Naive Bayes
    y_pred_nb = nb_classifier.predict(X_test_tfidf)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Naive Bayes Model Accuracy: {accuracy_nb:.2f}")

    # Initialize and train Support Vector Machine (SVM) classifier
    svm_classifier = SVC(kernel='linear', C=1.0)  # Adjust C for tuning
    svm_classifier.fit(X_train_tfidf, y_train)

    # Predict on the test data with SVM
    y_pred_svm = svm_classifier.predict(X_test_tfidf)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Model Accuracy: {accuracy_svm:.2f}")

    # Display classification reports
    print("\nNaive Bayes Classification Report:")
    print(classification_report(y_test, y_pred_nb, zero_division=1))

    print("\nSVM Classification Report:")
    print(classification_report(y_test, y_pred_svm, zero_division=1))

except FileNotFoundError:
    print("The specified file path does not exist.")
except pd.errors.ParserError:
    print("Error parsing the CSV file.")
except KeyError as e:
    print(f"Column not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
