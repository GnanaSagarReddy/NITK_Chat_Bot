import json
from urllib import response

import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import pickle
import re

with open('intents.json') as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

num_classes = len(labels)
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)
vocab_size = 1000

embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
epochs = 500
history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)
model.save('chat_model')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('label_encoder.pickle', 'wb') as enc:
    pickle.dump(lbl_encoder, enc, protocol=pickle.HIGHEST_PROTOCOL)

def get_chatbot_response(inp):
# Load the trained model
    model = keras.models.load_model('chat_model')

def get_chatbot_response(inp):
    # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                                      truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])

def send_message():
    user_input = user_input_text.get("1.0", "end-1c")
    if user_input.strip() != "":
        chat_text.configure(state=tk.NORMAL)
        chat_text.insert(tk.END, "You: " + user_input + "\n")
        chat_text.configure(state=tk.DISABLED)

        chatbot_response = get_chatbot_response(user_input)

        chat_text.configure(state=tk.NORMAL)
        chat_text.insert(tk.END, "ChatBot: " + chatbot_response + "\n")
        chat_text.configure(state=tk.DISABLED)

        user_input_text.delete("1.0", tk.END)

def quit_chat():
    if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
        window.destroy()

window = tk.Tk()
window.title("ChatBot")
window.geometry("500x500")

# Create the chat text display
chat_text = scrolledtext.ScrolledText(window, width=60, height=20)
chat_text.configure(state=tk.DISABLED)
chat_text.pack(padx=10, pady=10)

# Create the user input text box
user_input_text = tk.Text(window, width=50, height=3)
user_input_text.pack(padx=10)

# Create the send message button
send_button = tk.Button(window, text="Send", command=send_message)
send_button.pack(padx=10, pady=5)

# Create the quit button
quit_button = tk.Button(window, text="Quit", command=quit_chat)
quit_button.pack(padx=10, pady=5)

# Start the main loop
window.mainloop()






