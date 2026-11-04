# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 07:19:15 2024

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 04:25:27 2024
@author: Administrator
"""
# Naive Bayes, SVM, and Enhanced TF-IDF Techniques
# Bangash PhD Scholar 

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Specify the path to your CSV file
file_path = r"path"

# Load the CSV file and suppress intermediate outputs
try:
    df = pd.read_csv(file_path, encoding='gb18030')
    print("CSV file loaded successfully.")
    
    # Assuming 'answer' is the text column and 'department' is the label column
    X = df['answer']  # Features (text data)
    y = df['department']  # Labels (classification target)
    
    # Filter out rare classes with fewer than 10 samples
    min_samples = 10
    rare_classes = y.value_counts()[y.value_counts() < min_samples].index
    df = df[~df['department'].isin(rare_classes)]
    X = df['answer']
    y = df['department']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF with increased max features
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Apply SMOTE to balance classes for Naive Bayes
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

    # Naive Bayes with hyperparameter tuning
    nb_classifier = MultinomialNB()
    nb_params = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
    nb_search = RandomizedSearchCV(nb_classifier, nb_params, n_iter=5, scoring='accuracy', cv=3, random_state=42)
    nb_search.fit(X_train_resampled, y_train_resampled)
    print(f"Best Naive Bayes alpha: {nb_search.best_params_['alpha']}")

    # Predict and evaluate Naive Bayes model
    y_pred_nb = nb_search.predict(X_test_tfidf)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Naive Bayes Model Accuracy: {accuracy_nb:.2f}")
    print("\nNaive Bayes Classification Report:")
    print(classification_report(y_test, y_pred_nb, zero_division=1))

except FileNotFoundError:
    print("The specified file path does not exist.")
except pd.errors.ParserError:
    print("Error parsing the CSV file.")
except KeyError as e:
    print(f"Column not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
