# Импортируем библиотеки
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
# Загружаем датасет
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, 'content.json'), 'rt', encoding='utf-8') as jsonfile:
    data = json.load(jsonfile)

# Подготовка датасета
tags = []
inputs = []
responses = {}

for intent in data['intents']:
    responses[intent['tag']] = intent['responses']
    for line in intent['input']:
        inputs.append(line)
        tags.append(intent['tag'])

# Преобразование
data_df = pd.DataFrame({"inputs": inputs, "tags": tags})

# Предварительная обработка текстовых данных: удаление знаков препинания и строчных букв
data_df['inputs'] = data_df['inputs'].apply(lambda wrd: ''.join([l.lower() for l in wrd if l not in string.punctuation]))

# Токенизация текстовых данных
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data_df['inputs'])
x_train = tokenizer.texts_to_sequences(data_df['inputs'])
x_train = pad_sequences(x_train)

# Кодировка выходных меток
le = LabelEncoder()
y_train = le.fit_transform(data_df['tags'])

# Определить параметры модели
input_shape = x_train.shape[1]
print(input_shape)
vocabulary_size = len(tokenizer.word_index) + 1
print("Число уникальных слов:", vocabulary_size)
output_length = len(le.classes_)
print("output length:", output_length)

# Построение модели
inputs_layer = Input(shape=(input_shape,))
x = Embedding(vocabulary_size, 10)(inputs_layer)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(inputs=inputs_layer, outputs=x)

# Составить модель
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Обучить модель
history = model.fit(x_train, y_train, epochs=200, batch_size=8, verbose=1)


plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.legend()
plt.show()

# Функциональность чат-бота
def chat():
    print("Start talking to the Blood-donor_bot (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Blood-donor_bot: До свидания!")
            break

        # Предварительная обработка пользовательского ввода
        user_input = ''.join([char.lower() for char in user_input if char not in string.punctuation])
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=input_shape)

        # Предсказание ответа
        prediction = model.predict(padded_sequence, verbose=0)
        predicted_tag = le.inverse_transform([np.argmax(prediction)])[0]

        # Выбор случайного ответа из соответствующего тега
        bot_response = random.choice(responses[predicted_tag])
        print(f"Blood-donor_bot: {bot_response}")

# Чат-бот
if __name__ == "__main__":
    chat()

