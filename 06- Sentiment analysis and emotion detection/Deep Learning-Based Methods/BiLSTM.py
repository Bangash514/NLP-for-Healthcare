# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:09:41 2024

@author: Administrator
"""

#BiLSTM

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        self.u = self.add_weight(name="context_vector", shape=(input_shape[-1],), initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=[2, 0]) + self.b)
        score = tf.tensordot(score, self.u, axes=[2, 0])
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

# Load your data
file_path = r"\外科5-14000.csv"
df = pd.read_csv(file_path, encoding='gb18030')
print("CSV file loaded successfully.")

# Assuming 'answer' is the text column and 'department' is the label column
X = df['answer'].values  # Features (text data)
y = df['department'].values  # Labels (classification target)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Compute class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights_dict = dict(enumerate(class_weights))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Text Tokenization
max_words = 10000  # Vocabulary size
max_len = 100      # Maximum sequence length
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Load GloVe embeddings
embedding_index = {}

# Use a simplified path for GloVe
glove_path = "C:/glove.6B.100d.txt"  # Change this path if necessary

# Verify GloVe file existence
if os.path.isfile(glove_path):
    print("GloVe file found. Loading embeddings...")
    try:
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefficients = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefficients
        print("GloVe embeddings loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the GloVe file: {e}")
else:
    raise FileNotFoundError(f"GloVe file not found at {glove_path}. Please check the path.")

# Prepare the embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Build the BiLSTM model with Attention
input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len, 
                            weights=[embedding_matrix], trainable=False)(input_layer)
bilstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
dropout_layer = Dropout(0.5)(bilstm_layer)
attention_layer = AttentionLayer()(dropout_layer)
output_layer = Dense(y_categorical.shape[1], activation='softmax')(attention_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with class weights
history = model.fit(X_train_pad, y_train, epochs=30, batch_size=32, validation_split=0.2,
                    class_weight=class_weights_dict, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Get predicted classes
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Display the classification report with all possible labels
print(classification_report(
    y_true_classes,
    y_pred_classes,
    labels=np.arange(len(label_encoder.classes_)),  # Ensure all classes are included
    target_names=label_encoder.classes_,
    zero_division=1
))
