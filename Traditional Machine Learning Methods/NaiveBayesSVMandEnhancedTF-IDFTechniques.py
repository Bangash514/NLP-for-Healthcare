# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 04:25:27 2024
@author: Administrator
"""
# Naive Bayes, SVM, and Enhanced TF-IDF Techniques
# Bangash PhD Scholar 

import time
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Specify the path to your CSV file
file_path = r"data/your_file.csv"

# Load the CSV file
try:
    start_time = time.time()  # Start time for the entire script

    df = pd.read_csv(file_path, encoding='gb18030')
    print("CSV file loaded successfully.")
    
    # Assuming 'answer' is the text column and 'department' is the label column
    X = df['answer']  # Features (text data)
    y = df['department']  # Labels (classification target)
    
    # Optionally, remove classes with very few samples (e.g., less than 10) to avoid instability
    min_samples = 10
    rare_classes = y.value_counts()[y.value_counts() < min_samples].index
    df = df[~df['department'].isin(rare_classes)]
    X = df['answer']
    y = df['department']
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data using TF-IDF with increased max features
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Naive Bayes Model (without SVD)
    smote_nb = SMOTE(random_state=42, k_neighbors=1)
    X_train_resampled_nb, y_train_resampled_nb = smote_nb.fit_resample(X_train_tfidf, y_train)

    nb_classifier = MultinomialNB()
    nb_params = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
    nb_search = RandomizedSearchCV(nb_classifier, nb_params, n_iter=5, scoring='accuracy', cv=3, random_state=42)
    nb_search.fit(X_train_resampled_nb, y_train_resampled_nb)
    print(f"Best Naive Bayes alpha: {nb_search.best_params_['alpha']}")

    # Predict and evaluate Naive Bayes model
    y_pred_nb = nb_search.predict(X_test_tfidf)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Naive Bayes Model Accuracy: {accuracy_nb:.2f}")
    print("\nNaive Bayes Classification Report:")
    print(classification_report(y_test, y_pred_nb, zero_division=1))

    # SVM Model (with SVD and MinMaxScaler)
    svd = TruncatedSVD(n_components=300, random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)
    
    # Normalize data to non-negative values for compatibility with SVM
    scaler = MinMaxScaler()
    X_train_svd = scaler.fit_transform(X_train_svd)
    X_test_svd = scaler.transform(X_test_svd)
    
    smote_svm = SMOTE(random_state=42, k_neighbors=1)
    X_train_resampled_svm, y_train_resampled_svm = smote_svm.fit_resample(X_train_svd, y_train)

    svm_classifier = SVC(kernel='linear')
    svm_params = {'C': [0.01, 0.1, 1, 10, 100]}
    svm_search = RandomizedSearchCV(svm_classifier, svm_params, n_iter=5, scoring='accuracy', cv=3, random_state=42)
    svm_search.fit(X_train_resampled_svm, y_train_resampled_svm)
    print(f"Best SVM C: {svm_search.best_params_['C']}")

    # Predict and evaluate SVM model
    y_pred_svm = svm_search.predict(X_test_svd)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Model Accuracy: {accuracy_svm:.2f}")
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
finally:
    end_time = time.time()  # End time for the entire script
    runtime = end_time - start_time
