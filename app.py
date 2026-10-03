import json
import os
import re
import shutil

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json

st.set_page_config(
    page_title="English → French Translator",
    page_icon="🇫🇷",
    layout="wide",
)

# =============================================================
# Files (change these if yours are different)
# =============================================================
MODEL_DIR = "saved_model"
MAX_CHARS = 300

# Shown in the "Model performance" section.
# TODO: put the final validation loss of the run you exported here
# (it comes from the training log, e.g. "Epoch 6 | Val Loss: 0.7364").
VALIDATION_LOSS = "0.74"

EXAMPLES = [
    "I will go to school",
    "I want to go to school",
    "Where are you?",
    "I went to class",
]


# =============================================================
# Model classes (must be identical to the training notebook)
# =============================================================
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


# =============================================================
# Text helpers (same cleaning as training)
# =============================================================
def clean(s):
    s = str(s).lower().strip()
    s = re.sub(r"([.!?,¿])", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detokenize(s):
    s = re.sub(r"\s+([.,])", r"\1", s)  # French keeps a space before ? and !
    return s[:1].upper() + s[1:]


# =============================================================
# Load model, tokenizers and config
# =============================================================
def weights_path(name):
    """Keras 3 only reads '<name>.weights.h5'. If the file was saved or renamed
    as '<name>_weights.h5', make a correctly named copy."""
    good = f"{MODEL_DIR}/{name}.weights.h5"
    old = f"{MODEL_DIR}/{name}_weights.h5"
    if not os.path.exists(good) and os.path.exists(old):
        shutil.copy(old, good)
    return good


@st.cache_resource(show_spinner="Loading model...")
def load_assets():
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
    n_params = encoder.count_params() + decoder.count_params()
    return cfg, eng_tok, fr_tok, encoder, decoder, n_params


try:
    cfg, eng_tok, fr_tok, encoder, decoder, N_PARAMS = load_assets()
except Exception as exc:
    st.error(
        "The model could not be loaded. Check that the saved_model folder holds "
        "config.json, eng_tokenizer.json, fr_tokenizer.json, encoder.weights.h5 and "
        "decoder.weights.h5, and that requirements.txt matches the TensorFlow "
        "version used for training."
    )
    st.exception(exc)
    st.stop()

MAX_ENG_LEN = cfg["MAX_ENG_LEN"]
MAX_FR_LEN = cfg["MAX_FR_LEN"]
OOV_ENG = eng_tok.word_index.get(eng_tok.oov_token, 1)
OOV_FR = fr_tok.word_index.get(fr_tok.oov_token)


# =============================================================
# Translation
# =============================================================
def translate(sentence):
    tokens = clean(sentence).split()
    ids = eng_tok.texts_to_sequences([" ".join(tokens)])[0]
    n_words = len(ids)
    n_unknown = ids.count(OOV_ENG)
    truncated = n_words > MAX_ENG_LEN
    tokens, ids = tokens[:MAX_ENG_LEN], ids[:MAX_ENG_LEN]

    # IMPORTANT: no padding at inference. The attention layer has no source mask,
    # so padded positions would soak up attention weight and wreck short inputs.
    encoder_outputs, decoder_hidden = encoder(tf.constant([ids], dtype=tf.int32))

    start_token = fr_tok.word_index["<start>"]
    end_token = fr_tok.word_index["<end>"]
    decoder_input_word = tf.expand_dims([start_token], 0)

    rows, seen_bigrams, prev_word, stopped_early = [], set(), "<start>", False

    for _ in range(MAX_FR_LEN):
        predictions, decoder_hidden, attention = decoder(
            decoder_input_word, decoder_hidden, encoder_outputs
        )
        logits = predictions[0].numpy()
        if OOV_FR is not None:
            logits[OOV_FR] = -1e9

        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        predicted_id = int(np.argmax(logits))
        if predicted_id == end_token:
            break

        word = fr_tok.index_word.get(predicted_id, "").replace("’", "'")

        # the model is looping: treat a repeated word pair as the end
        if (prev_word, word) in seen_bigrams:
            stopped_early = True
            break
        seen_bigrams.add((prev_word, word))
        prev_word = word

        attn = attention.numpy().reshape(-1)
        rows.append(
            {
                "French word": word,
                "Looked at (English)": tokens[int(attn.argmax())],
                "Attention (%)": float(attn.max() * 100),
                "Confidence (%)": float(probs[predicted_id] * 100),
            }
        )
        decoder_input_word = tf.expand_dims([predicted_id], 0)

    words = [r["French word"] for r in rows]
    return {
        "text": sentence,
        "translation": detokenize(" ".join(words)),
        "rows": rows,
        "avg_conf": float(np.mean([r["Confidence (%)"] for r in rows])) if rows else 0.0,
        "n_words": n_words,
        "n_unknown": n_unknown,
        "truncated": truncated,
        "stopped_early": stopped_early,
    }


def use_example(text):
    st.session_state["text"] = text


# =============================================================
# Session state
# =============================================================
st.session_state.setdefault("text", "")
st.session_state.setdefault("result", None)

# =============================================================
# Sidebar
# =============================================================
with st.sidebar:
    st.header("How to use")
    st.markdown(
        """
        1. Type an English sentence, or click one of the examples.
        2. Press **Translate**.
        3. Read the French translation. The table beside it shows which English
           word each French word was built from.
        """
    )
    st.header("How it works")
    st.markdown(
        f"""
        1. The sentence is lowercased, punctuation is separated, and each word is
           turned into a number (a Keras tokenizer with a
           {cfg["ENG_VOCAB_SIZE"]:,}-word English vocabulary).
        2. A **GRU encoder** reads the English words.
        3. A **GRU decoder** writes French one word at a time. At every step,
           **Bahdanau attention** lets it look back at the English words that
           matter most.
        4. It stops when it writes the end token (or starts repeating itself).
        """
    )
    st.subheader("Good to know")
    st.markdown(
        f"""
        - Short, simple, correctly spelled sentences work best.
        - Only the first {MAX_ENG_LEN} words are read.
        - Words the model never saw in training are treated as unknown.
        - Short phrases without a verb (like "near the sea") can be mistranslated.
        - This is a small model trained from scratch, not a replacement for a
          professional translator.
        """
    )

# =============================================================
# Header and input
# =============================================================
st.title("🇫🇷 English → French Translator")
st.caption(
    "Type an English sentence and a GRU encoder–decoder with attention "
    "translates it into French."
)
st.divider()

st.text_area(
    "English sentence",
    key="text",
    placeholder="Example: I will go to school",
    height=110,
    max_chars=MAX_CHARS,
)

st.caption("Try an example:")
example_cols = st.columns(len(EXAMPLES))
for i, (col, text) in enumerate(zip(example_cols, EXAMPLES)):
    col.button(text, on_click=use_example, args=(text,), key=f"example_{i}")

if st.button("🔍 Translate", type="primary"):
    cleaned = " ".join(st.session_state["text"].split())
    if cleaned == "":
        st.warning("Please enter a sentence first.")
    else:
        with st.spinner("Translating..."):
            st.session_state["result"] = translate(cleaned)

# =============================================================
# Results
# =============================================================
result = st.session_state["result"]

if result is None:
    st.info("The translation will appear here after you press Translate.")
else:
    st.divider()

    known_words = result["n_words"] - result["n_unknown"]
    if result["n_words"] == 0 or known_words == 0:
        st.warning(
            "None of the words in this sentence are in the model's vocabulary, so "
            "the translation below is not reliable. Try a simple English sentence."
        )
    elif result["avg_conf"] < 50:
        st.info(
            "The model was unsure about several words, so treat this translation "
            "with care."
        )

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.subheader("Translation")
        st.success(f"🇫🇷  **{result['translation'] or '(no output)'}**")

        m1, m2, m3 = st.columns(3)
        m1.metric("English words read", result["n_words"])
        m2.metric("Unknown words", result["n_unknown"])
        m3.metric("Average word confidence", f"{result['avg_conf']:.1f}%")

        if result["truncated"]:
            st.caption(f"Only the first {MAX_ENG_LEN} words were used.")
        if result["stopped_early"]:
            st.caption("The model started repeating itself, so the output was cut there.")

    with right:
        st.subheader("Word by word")
        df = pd.DataFrame(result["rows"])
        if df.empty:
            st.write("No words were generated.")
        else:
            st.dataframe(
                df.style.format({"Attention (%)": "{:.0f}", "Confidence (%)": "{:.0f}"}),
                hide_index=True,
            )
            st.caption(
                "“Looked at” is the English word the decoder paid most attention "
                "to when it wrote that French word."
            )

# =============================================================
# Model details
# =============================================================
st.divider()
st.subheader("Model performance")
st.caption(
    "A translator has no star-style accuracy. Quality is tracked with validation "
    "loss: the average cross-entropy per French word on sentences it never trained "
    "on (lower is better)."
)

metrics = {
    "Validation loss": VALIDATION_LOSS,
    "Parameters": f"{N_PARAMS / 1e6:.1f} M",
    "English vocabulary": f"{cfg['ENG_VOCAB_SIZE']:,} words",
    "French vocabulary": f"{cfg['FR_VOCAB_SIZE']:,} words",
    "Max sentence length": f"{MAX_ENG_LEN} words",
}
cols = st.columns(len(metrics))
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, value)

with st.expander("Model architecture"):
    st.code(
        f"ENCODER\n"
        f"  Embedding ({cfg['ENG_VOCAB_SIZE']:,} words x {cfg['EMBEDDING_DIM']})\n"
        f"  GRU ({cfg['HIDDEN_UNITS']} units, returns sequence + state)\n"
        f"\n"
        f"ATTENTION (Bahdanau / additive)\n"
        f"  Dense ({cfg['HIDDEN_UNITS']}) on encoder outputs\n"
        f"  + Dense ({cfg['HIDDEN_UNITS']}) on decoder state -> tanh -> Dense (1)\n"
        f"  Softmax over the English words -> context vector\n"
        f"\n"
        f"DECODER (one French word per step)\n"
        f"  Embedding ({cfg['FR_VOCAB_SIZE']:,} words x {cfg['EMBEDDING_DIM']})\n"
        f"  Concatenate [context vector, word embedding]\n"
        f"  GRU ({cfg['HIDDEN_UNITS']} units)\n"
        f"  Dense ({cfg['FR_VOCAB_SIZE']:,}) -> next French word",
        language="text",
    )

# =============================================================
# Footer
# =============================================================
st.divider()
st.markdown(
    "Developed by **Karan Gojiya** | [GitHub](https://github.com/KaranGojiya) | "
    "[LinkedIn](https://www.linkedin.com/in/karan-gojiya)"
)
