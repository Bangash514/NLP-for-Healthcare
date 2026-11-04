

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# Specify the path to your CSV file
file_path = r"\外科5-14000.csv"

# Load the CSV file
try:
    df = pd.read_csv(file_path, encoding='gb18030')
    print("CSV file loaded successfully.")
    
    # Display the first few rows of the data
    print(df.head())
    
    # Assuming 'answer' is the text column and 'department' is the label column
    X = df['answer']  # Features (text data)
    y = df['department']  # Labels (classification target)
    
    # Combine features and labels into a single DataFrame
    df_balanced = df[['answer', 'department']].copy()

    # Identify the largest and smallest classes for resampling
    largest_class = df_balanced['department'].value_counts().idxmax()
    smallest_class = df_balanced['department'].value_counts().idxmin()
    
    # Separate the majority and minority classes
    df_majority = df_balanced[df_balanced['department'] == largest_class]
    df_minority = df_balanced[df_balanced['department'] == smallest_class]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                       replace=False,  # sample without replacement
                                       n_samples=len(df_minority)*10,  # Downsample to 10 times the size of the smallest class
                                       random_state=42)

    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,  # sample with replacement
                                     n_samples=50,  # Set to at least 50 samples
                                     random_state=42)

    # Combine the balanced classes with the original data
    df_balanced = pd.concat([df_balanced, df_majority_downsampled, df_minority_upsampled])

    # Split the balanced dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df_balanced['answer'], df_balanced['department'], test_size=0.2, random_state=42)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Initialize and train Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)

    # Predict on the test data
    y_pred = nb_classifier.predict(X_test_tfidf)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Display classification report with zero_division parameter to handle undefined metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

except FileNotFoundError:
    print("The specified file path does not exist.")
except pd.errors.ParserError:
    print("Error parsing the CSV file.")
except KeyError as e:
    print(f"Column not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
