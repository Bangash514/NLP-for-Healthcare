# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:20:59 2024

@author: Administrator
"""
#Bangash PhD Scholar
#BiLSTM and Attention Mechanism

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class TextPreprocessor:
    def __init__(self, max_vocab_size=10000, max_seq_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        self.label_encoder = LabelEncoder()

    def build_vocab(self, texts):
        for text in texts:
            words = jieba.lcut(str(text))
            self.word_freq.update(words)

        most_common = self.word_freq.most_common(self.max_vocab_size - 2)
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode_text(self, text):
        words = jieba.lcut(str(text))
        encoded = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        if len(encoded) > self.max_seq_length:
            encoded = encoded[:self.max_seq_length]
        return encoded

class HealthcareDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded_text = self.preprocessor.encode_text(text)
        return torch.tensor(encoded_text), torch.tensor(label, dtype=torch.long)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, lstm_output):
        attention_scores = self.attention_weights(lstm_output).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        return context, attention_weights

class BiLSTMWithAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx):
        super(BiLSTMWithAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True, 
                              batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), 
                                                            batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.bilstm(packed_embedded)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        context, attention_weights = self.attention(lstm_output)
        return self.fc(self.dropout(context))

def collate_batch(batch):
    texts, labels = zip(*batch)
    text_lengths = torch.tensor([len(text) for text in texts])
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded_texts, torch.stack(labels), text_lengths

class HealthcareNLPTrainer:
    def __init__(self, file_path, max_vocab_size=10000, max_seq_length=100):
        self.file_path = file_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = TextPreprocessor(max_vocab_size, max_seq_length)
        self.history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}

    def load_data(self):
        print("Loading and preprocessing data...")
        df = pd.read_csv(self.file_path, encoding='gb18030').sample(1000)  # Load only 1000 rows for testing
        print("Column names in the dataset:", df.columns)
        
        # Use 'ask' for text and 'department' for labels
        texts = df['ask'].values  # Text data
        labels = df['department'].values  # Labels

        # Encode labels
        labels = self.preprocessor.label_encoder.fit_transform(labels)

        # Build vocabulary
        self.preprocessor.build_vocab(texts)

        # Split data
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42)

        # Create datasets with reduced batch size
        train_dataset = HealthcareDataset(train_texts, train_labels, self.preprocessor)
        valid_dataset = HealthcareDataset(valid_texts, valid_labels, self.preprocessor)

        self.train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_batch, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=16, collate_fn=collate_batch)

        return len(self.preprocessor.word2idx), len(self.preprocessor.label_encoder.classes_)

    def train_model(self, embedding_dim=50, hidden_dim=128, n_layers=1, dropout=0.3, n_epochs=5):
        vocab_size, output_dim = self.load_data()
        model = BiLSTMWithAttentionModel(vocab_size, embedding_dim, hidden_dim, output_dim, 
                                         n_layers, dropout, pad_idx=0).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        print(f"Training on {self.device}")
        
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{n_epochs}'):
                texts, labels, lengths = [x.to(self.device) for x in batch]
                optimizer.zero_grad()
                predictions = model(texts, lengths)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += (predictions.argmax(1) == labels).sum().item()
                train_total += len(labels)
            
            model.eval()
            valid_loss = 0
            valid_correct = 0
            valid_total = 0
            
            with torch.no_grad():
                for batch in self.valid_loader:
                    texts, labels, lengths = [x.to(self.device) for x in batch]
                    predictions = model(texts, lengths)
                    loss = criterion(predictions, labels)
                    
                    valid_loss += loss.item()
                    valid_correct += (predictions.argmax(1) == labels).sum().item()
                    valid_total += len(labels)
            
            train_loss /= len(self.train_loader)
            valid_loss /= len(self.valid_loader)
            train_acc = train_correct / train_total
            valid_acc = valid_correct / valid_total
            
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)
            self.history['train_acc'].append(train_acc)
            self.history['valid_acc'].append(valid_acc)
            
            print(f'\nEpoch: {epoch+1}')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}')
        
        return model

    def plot_training_history(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['valid_loss'], label='Valid Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['valid_acc'], label='Valid Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Define the main function at the end of the code
def main():
    file_path = r"E:\PhD 2022-26\西安交通大学\2024-25\NLPForHealthcare\DataPath\外科5-14000.csv"
    trainer = HealthcareNLPTrainer(file_path)
    model = trainer.train_model()
    trainer.plot_training_history()

if __name__ == "__main__":
    main()
