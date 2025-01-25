# Importing necessary libraries
import tensorflow as tf
import sys
import os
import numpy as np
import pandas as pd
import json
import string
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, 'content.json'), 'rt', encoding='utf-8') as jsonfile:
    data = json.load(jsonfile)

# Preparing the dataset
tags = []
inputs = []
responses = {}

for intent in data['intents']:
    responses[intent['tag']] = intent['responses']
    for line in intent['input']:
        inputs.append(line)
        tags.append(intent['tag'])

# Convert to DataFrame
data_df = pd.DataFrame({"inputs": inputs, "tags": tags})

# Preprocess text data: remove punctuation and lowercase
data_df['inputs'] = data_df['inputs'].apply(lambda wrd: ''.join([l.lower() for l in wrd if l not in string.punctuation]))

# Tokenize text data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data_df['inputs'])
x_train = tokenizer.texts_to_sequences(data_df['inputs'])
x_train = pad_sequences(x_train)

# Encode output labels
le = LabelEncoder()
y_train = le.fit_transform(data_df['tags'])

# Define model parameters
input_shape = x_train.shape[1]
vocabulary_size = len(tokenizer.word_index) + 1
output_length = len(le.classes_)

# Build the model
inputs_layer = Input(shape=(input_shape,))
x = Embedding(vocabulary_size, 10)(inputs_layer)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(inputs=inputs_layer, outputs=x)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(x_train, y_train, epochs=200, batch_size=8, verbose=1)

# Plot training history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.legend()
plt.show()

# Chat functionality
def chat():
    print("Start talking to the Blood-donor_bot (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Blood-donor_bot: До свидания!")
            break

        # Preprocess user input
        user_input = ''.join([char.lower() for char in user_input if char not in string.punctuation])
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=input_shape)

        # Predict response
        prediction = model.predict(padded_sequence, verbose=0)
        predicted_tag = le.inverse_transform([np.argmax(prediction)])[0]

        # Select random response from matched tag
        bot_response = random.choice(responses[predicted_tag])
        print(f"Blood-donor_bot: {bot_response}")

# Start the chatbot
if __name__ == "__main__":
    chat()
