# English → French Neural Machine Translator

A sequence-to-sequence translator built with **TensorFlow/Keras**, using a GRU encoder–decoder with **Bahdanau (additive) attention**, served through a **Streamlit** web app.

## Demo examples

| English | French (model output) |
|---|---|
| I will go to school | J'irai à l'école. |
| I went to class | Je suis allé à la classe. |
| You want to go out | Tu veux sortir. |
| I love you | Je t'aime. |

## Model

| Component | Details |
|---|---|
| Encoder | Embedding (256) → GRU (512), returns all outputs and final state |
| Attention | Bahdanau additive attention over encoder outputs |
| Decoder | Embedding (256) → concat(context, embedding) → GRU (512) → Dense(vocab) |
| Training | Teacher forcing, Adam, masked sparse categorical cross-entropy (padding ignored) |
| Inference | Greedy decoding, starts at `<start>`, stops at `<end>` |

**Data**
- 150,000 English–French sentence pairs sampled from `english_french.csv`
- Filtered to English sentences of 1–10 words and French sentences of 1–12 words
- 80/20 train/validation split (120,000 / 30,000)
- Lowercased, with punctuation split into separate tokens
- Vocabularies built with the Keras `Tokenizer` (English and French separately)

**Result** (6 epochs, batch size 64)

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 1.9779 | 1.5587 |
| 3 | 0.8027 | 0.8424 |
| 6 | 0.3582 | 0.7364 |

## Project structure

```
project/
├── app.py                  # Streamlit app (model classes + inference + UI)
├── requirements.txt
├── README.md
├── English_to_French.ipynb # Training notebook
└── saved_model/
    ├── encoder.weights.h5
    ├── decoder.weights.h5
    ├── eng_tokenizer.json
    ├── fr_tokenizer.json
    └── config.json
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Important:** use the same TensorFlow/Keras version that you trained with. Weights saved with Keras 3 (`.weights.h5`) need Keras 3 to load. In Colab, run `print(tf.__version__)` and pin that exact version in `requirements.txt` (for example `tensorflow==2.19.0`).

## Exporting the trained model from the notebook

Run this at the end of the training notebook, then unzip the download into `saved_model/`:

```python
import os, json, shutil

os.makedirs("saved_model", exist_ok=True)

encoder.save_weights("saved_model/encoder.weights.h5")
decoder.save_weights("saved_model/decoder.weights.h5")

with open("saved_model/eng_tokenizer.json", "w") as f:
    f.write(eng_tokenizer.to_json())
with open("saved_model/fr_tokenizer.json", "w") as f:
    f.write(fr_tokenizer.to_json())

config = {
    "ENG_VOCAB_SIZE": ENG_VOCAB_SIZE,
    "FR_VOCAB_SIZE": FR_VOCAB_SIZE,
    "EMBEDDING_DIM": EMBEDDING_DIM,
    "HIDDEN_UNITS": HIDDEN_UNITS,
    "MAX_ENG_LEN": MAX_ENG_LEN,
    "MAX_FR_LEN": MAX_FR_LEN,
}
with open("saved_model/config.json", "w") as f:
    json.dump(config, f)

shutil.make_archive("saved_model", "zip", "saved_model")

from google.colab import files
files.download("saved_model.zip")
```

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Important notes

- **Training and inference must use the same preprocessing.** `app.py` uses the same `clean()` function as the notebook (lowercase, punctuation split into tokens). If you change it in one place, change it in the other and retrain.
- **Model classes must match the notebook.** If you add dropout, an attention mask or change layer sizes, update `Encoder`, `BahdanauAttention` and `Decoder` in `app.py` and re-export the weights.
- **Subclassed Keras models must be built before loading weights.** `load_artifacts()` handles this by calling both models once on dummy input.

## Limitations

- Trained on short sentences: input is limited to **10 English words**, and longer input is truncated.
- Works best on complete, simple sentences. Short noun phrases (for example "Near the sea") can be mistranslated because they are rare in the training data.
- Words not seen in training become `<OOV>`, which lowers translation quality.
- Greedy decoding only, with no beam search.
- Output is not a substitute for a professional translation.

## Possible improvements

- Beam search decoding
- Attention masking for padded positions
- Dropout and label smoothing to reduce overfitting
- BLEU score evaluation on the validation set
- Subword tokenization (BPE/SentencePiece) to handle rare words
- Attention heatmap visualization in the Streamlit UI

## Tech stack

Python · TensorFlow/Keras · Streamlit · pandas · scikit-learn
