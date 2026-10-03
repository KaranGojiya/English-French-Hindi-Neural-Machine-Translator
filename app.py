import json
import os
import re
import shutil

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json

MODEL_DIR = "saved_model"


# ---------- Model classes (must be identical to the notebook) ----------
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True
        )
        self.gru = tf.keras.layers.GRU(
            hidden_units, return_sequences=True, return_state=True
        )

    def call(self, x):
        x = self.embedding(x)
        encoder_outputs, encoder_state = self.gru(x)
        return encoder_outputs, encoder_state


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(hidden_units)
        self.W2 = tf.keras.layers.Dense(hidden_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, decoder_hidden, encoder_outputs):
        decoder_hidden_with_time_axis = tf.expand_dims(decoder_hidden, 1)
        score = self.V(
            tf.nn.tanh(
                self.W1(encoder_outputs) + self.W2(decoder_hidden_with_time_axis)
            )
        )
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * encoder_outputs, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True
        )
        self.attention = BahdanauAttention(hidden_units)
        self.gru = tf.keras.layers.GRU(
            hidden_units, return_sequences=True, return_state=True
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, decoder_hidden, encoder_outputs):
        context_vector, attention_weights = self.attention(
            decoder_hidden, encoder_outputs
        )
        x = self.embedding(x)
        context_vector = tf.expand_dims(context_vector, 1)
        x = tf.concat([context_vector, x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        prediction = self.fc(output)
        return prediction, state, attention_weights


# ---------- Text helpers (same cleaning as training) ----------
def clean(s):
    s = str(s).lower().strip()
    s = re.sub(r"([.!?,¿])", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detokenize(s):
    s = re.sub(r"\s+([.!?,])", r"\1", s)
    return s[:1].upper() + s[1:]


# ---------- Load everything once ----------
def weights_path(name):
    """Keras 3 only reads '<name>.weights.h5'. If the file was saved/renamed as
    '<name>_weights.h5', make a correctly named copy."""
    good = f"{MODEL_DIR}/{name}.weights.h5"
    old = f"{MODEL_DIR}/{name}_weights.h5"
    if not os.path.exists(good) and os.path.exists(old):
        shutil.copy(old, good)
    return good



@st.cache_resource
def load_artifacts():
    with open(f"{MODEL_DIR}/config.json") as f:
        cfg = json.load(f)
    with open(f"{MODEL_DIR}/eng_tokenizer.json") as f:
        eng_tok = tokenizer_from_json(f.read())
    with open(f"{MODEL_DIR}/fr_tokenizer.json") as f:
        fr_tok = tokenizer_from_json(f.read())

    encoder = Encoder(cfg["ENG_VOCAB_SIZE"], cfg["EMBEDDING_DIM"], cfg["HIDDEN_UNITS"])
    decoder = Decoder(cfg["FR_VOCAB_SIZE"], cfg["EMBEDDING_DIM"], cfg["HIDDEN_UNITS"])

    # Subclassed models must be built (called once) before loading weights
    enc_out, enc_state = encoder(tf.zeros((1, cfg["MAX_ENG_LEN"]), dtype=tf.int32))
    decoder(tf.zeros((1, 1), dtype=tf.int32), enc_state, enc_out)

    encoder.load_weights(weights_path("encoder"))
    decoder.load_weights(weights_path("decoder"))
    return cfg, eng_tok, fr_tok, encoder, decoder


def translate(sentence, cfg, eng_tok, fr_tok, encoder, decoder):
    # IMPORTANT: no padding at inference. The attention layer has no source mask,
    # so padded positions would soak up attention weight and wreck short inputs.
    # Feeding only the real tokens gives attention nothing but real words.
    ids = eng_tok.texts_to_sequences([clean(sentence)])[0][: cfg["MAX_ENG_LEN"]]
    encoder_outputs, decoder_hidden = encoder(tf.constant([ids], dtype=tf.int32))

    start_token = fr_tok.word_index["<start>"]
    end_token = fr_tok.word_index["<end>"]
    decoder_input_word = tf.expand_dims([start_token], 0)

    result = []
    seen_bigrams = set()
    prev_word = "<start>"
    oov_id = fr_tok.word_index.get("<OOV>")

    for _ in range(cfg["MAX_FR_LEN"]):
        predictions, decoder_hidden, _ = decoder(
            decoder_input_word, decoder_hidden, encoder_outputs
        )
        logits = predictions[0].numpy()
        if oov_id is not None:
            logits[oov_id] = -1e9

        predicted_id = int(np.argmax(logits))
        if predicted_id == end_token:
            break

        word = fr_tok.index_word.get(predicted_id, "").replace("’", "'")

        # the model is looping: treat a repeated word pair as the end
        if (prev_word, word) in seen_bigrams:
            break
        seen_bigrams.add((prev_word, word))
        prev_word = word

        result.append(word)
        decoder_input_word = tf.expand_dims([predicted_id], 0)

    return detokenize(" ".join(result))


# ---------- UI ----------
st.set_page_config(page_title="English to French Translator", page_icon="🇫🇷")
st.title("English → French Translator")
st.caption("Encoder–Decoder · GRU · Bahdanau Attention (TensorFlow)")

cfg, eng_tok, fr_tok, encoder, decoder = load_artifacts()

text = st.text_input("Enter an English sentence", placeholder="I will go to school")

if st.button("Translate", type="primary") and text.strip():
    if len(text.split()) > cfg["MAX_ENG_LEN"]:
        st.warning(
            f"The model was trained on sentences up to {cfg['MAX_ENG_LEN']} words, "
            "so the end of your input will be cut off."
        )
    with st.spinner("Translating..."):
        output = translate(text, cfg, eng_tok, fr_tok, encoder, decoder)
    st.success(output)
