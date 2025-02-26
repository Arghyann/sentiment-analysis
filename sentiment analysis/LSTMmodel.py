from tokenise import tokenise
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.regularizers import l2

df=pd.read_csv("cleaned_IMDB_Dataset.csv")
padded_sequences, labels, tokenizer = tokenise(df, save_tokenizer=True)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)


embedding_dim = 128  # Size of word embeddings
lstm_units = 64      # Number of LSTM units
vocab_size = 20000   # Must match tokenizer num_words
maxlen = 300         # Same as in tokenization step

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    LSTM(lstm_units, return_sequences=True),
    Dropout(0.2),
    LSTM(lstm_units),
    Dropout(0.2),
    Dense(1, activation="sigmoid")  # Binary classification (positive/negative)
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

batch_size = 64
epochs = 10  # Adjust based on performance

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=batch_size, epochs=epochs, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

model.save("sentiment_model.keras")