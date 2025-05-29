import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

model=load_model("Predictive_typing_GRU.keras")
with open("max_len_seq.pkl","rb") as file:
    max_sequence_len=pickle.load(file)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_next_word(seed_text):
    seed_text = seed_text.lower().strip()
    tokens = tokenizer.texts_to_sequences([seed_text])[0]

    if len(tokens) == 0:
        return "[Unknown input]"

    # Wrap in a list to create batch of size 1
    padded = pad_sequences([tokens], maxlen=max_sequence_len-1, padding='pre')

    # Make prediction
    predicted_probs = model.predict(padded, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]

    return tokenizer.index_word.get(predicted_index, "[UNK]")

st.title("Predictive Typing Assistant")
seed_text=st.text_input("type your message","")

if st.button("predict next word"):
    if seed_text.strip():
        next_word=predict_next_word(seed_text)
        st.success(f"next word prediction **{next_word}**")
    else:
        st.warning("please type something else to get prediction")
