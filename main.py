import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

# Directories
DATA_DIR = './data'
MODELS_DIR = './models'

os.makedirs(MODELS_DIR, exist_ok=True)

# 1a. Traditional Model for Text Prediction: N-gram + Naive Bayes
def train_text_prediction_nb(data_path):
    with open(data_path, 'r') as f:
        corpus = [line.strip() for line in f.readlines() if line.strip()]

    # Prepare input-output pairs from corpus
    X_texts = []
    y_words = []
    for line in corpus:
        words = line.split()
        for i in range(1, len(words)):
            X_texts.append(' '.join(words[:i]))
            y_words.append(words[i])

    # Vectorize input text (n-gram)
    vectorizer = CountVectorizer(ngram_range=(1,2))
    X_vec = vectorizer.fit_transform(X_texts)

    # Label encode outputs (words)
    vocab = sorted(set(y_words))
    word_to_idx = {w:i for i,w in enumerate(vocab)}
    y_idx = np.array([word_to_idx[w] for w in y_words])

    # Train Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_vec, y_idx)

    # Save model, vectorizer and word_to_idx
    with open(os.path.join(MODELS_DIR, 'text_pred_nb.pkl'), 'wb') as f:
        pickle.dump((model, vectorizer, word_to_idx), f)

    print("Traditional NB text prediction model saved!")

# 1b. Neural Network Model for Text Prediction (your existing LSTM)
def train_text_prediction_lstm(data_path):
    with open(data_path, 'r') as f:
        corpus = [line.strip() for line in f.readlines() if line.strip()]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)
    vocab_size = len(tokenizer.word_index) + 1

    input_sequences = []
    for seq in sequences:
        for i in range(1, len(seq)):
            input_sequences.append(seq[:i+1])
    max_len = max(len(seq) for seq in input_sequences)
    input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)

    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_len-1),
        LSTM(100),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X, y, epochs=30, validation_split=0.2, callbacks=[early_stop], verbose=2)

    model.save(os.path.join(MODELS_DIR, 'lstm_model.h5'))
    with open(os.path.join(MODELS_DIR, 'lstm_tokenizer.pkl'), 'wb') as f:
        pickle.dump((tokenizer, max_len), f)

    print("LSTM text prediction model saved!")

# 2a. Traditional Model for Sentiment Classification (your existing Naive Bayes)
def train_sentiment_naive_bayes(data_path):
    df = pd.read_csv(data_path)
    X = df['text'].str.lower().str.strip().str.replace('[^\w\s]', '', regex=True)
    y = df['label'].astype(int)

    print("Sample texts:", X.head())
    print("Label distribution:\n", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    with open(os.path.join(MODELS_DIR, 'naive_bayes.pkl'), 'wb') as f:
        pickle.dump((model, vectorizer), f)

    y_pred = model.predict(X_test_vec)
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
    print("Naive Bayes sentiment classification model saved!")

# 2b. Neural Network Model for Sentiment Classification (LSTM)
def train_sentiment_lstm(data_path):
    df = pd.read_csv(data_path)
    X = df['text'].str.lower().str.strip().str.replace('[^\w\s]', '', regex=True)
    y = df['label'].astype(int)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    max_len = max(len(seq) for seq in sequences)
    X_padded = pad_sequences(sequences, maxlen=max_len, padding='post')

    y_cat = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X_padded, y_cat, test_size=0.2, random_state=42)

    vocab_size = len(tokenizer.word_index) + 1

    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_len),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=2)

    model.save(os.path.join(MODELS_DIR, 'sentiment_lstm.h5'))
    with open(os.path.join(MODELS_DIR, 'sentiment_tokenizer.pkl'), 'wb') as f:
        pickle.dump((tokenizer, max_len), f)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM Sentiment Classification Accuracy: {acc:.4f}")
    print("LSTM sentiment classification model saved!")

# Main Function
if __name__ == "__main__":
    # Text Prediction
    train_text_prediction_nb(os.path.join(DATA_DIR, 'text_prediction.txt'))  # Traditional
    train_text_prediction_lstm(os.path.join(DATA_DIR, 'text_prediction.txt'))  # Neural Network

    # Sentiment Classification
    train_sentiment_naive_bayes(os.path.join(DATA_DIR, 'sentiment_data.csv'))  # Traditional
    train_sentiment_lstm(os.path.join(DATA_DIR, 'sentiment_data.csv'))  # Neural Network
