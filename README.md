# English → French & Hindi Neural Machine Translator

A sequence-to-sequence translator built from scratch with **TensorFlow / Keras** (GRU encoder–decoder with **Bahdanau attention** and **beam search**), served through a **Streamlit** web app. Pick French or Hindi, type an English sentence, and see the translation together with a word-by-word view of which English word the model was looking at.

## Examples

Real outputs of the exported models (beam search, width 4).

| English | French | Hindi |
|---|---|---|
| I love you | Je t'aime. | मैं तुमसे प्यार करता हूँ |
| I will go to school | J'irai à l'école. | मैं स्कूल जाना चाहता हूँ ⚠️ |
| I am going to school | Je vais à l'école. | मैं स्कूल जा रहा हूँ |
| Where are you? | Où êtes-vous ? | तुम कहाँ हैं? ⚠️ |
| Where is the station? | Où est la gare ? | स्टेशन कहाँ है? |
| How are you? | Comment allez-vous ? | आप कैसे हैं? |
| Can you help me | Tu peux m'aider ? | क्या आप मुझे मदद कर सकते हैं |
| Who are you | Es-tu ? ⚠️ | आप कौन हैं |

**⚠️ Not a correct translation.**

## App features

- **Language switch** (French / Hindi). Each model is loaded only when it is used.
- **Beam search** (width 4) with a no-repeat-trigram rule.
- **Word-by-word table**: for every output word, the English word the decoder attended to most, the attention weight and the model's confidence.
- **Reliability warnings** for unknown vocabulary, low confidence and over-long input.
- **Model performance panel** that reads each model's own `config.json` (loss, BLEU, parameters, vocabulary sizes) and an architecture view.

## Model

Both languages use the same architecture and the same code (`app.py`).

| Component | Details |
|---|---|
| Encoder | Embedding (256) → GRU (512, returns all outputs and the final state) |
| Attention | Bahdanau (additive) attention over the encoder outputs, with a source mask for padding |
| Decoder | Embedding (256) → concat(context vector, embedding) → GRU (512) → Dense(vocabulary) |
| Regularisation | Dropout 0.3 on embeddings and decoder output, gradient clipping 1.0 |
| Training | Teacher forcing, Adam (lr 1e-3, halved when validation stalls), batch 128, early stopping (patience 3) with the best weights restored |
| Loss | Cross-entropy averaged over **real words only** (padding excluded) |
| Decoding | Beam search (width 4, length penalty 0.6), no padding at inference, `<OOV>` never output |

| | English → French | English → Hindi |
|---|---|---|
| English vocabulary | 14,216 | 15,000 (capped; covers 95.9% of word occurrences) |
| Target vocabulary | 27,130 | 20,000 (capped; covers 96.7% of word occurrences) |
| Max sentence length (source / target) | 15 / 17 | 25 / 26 |
| Parameters | 28.2 M | 22.9 M |

## Data and preprocessing (Hindi)

Source: `Dataset_English_Hindi.csv`.

| Step | Pairs |
|---|---|
| Raw rows | 130,476 |
| After removing missing values and duplicates | 127,375 |
| After filtering | **95,064** |
| Train / validation / test | 89,360 / 2,852 / 2,852 |

Cleaning and filtering:

- Unicode NFC normalisation, quote normalisation, removal of invisible joiners, Devanagari digits → ASCII digits.
- Sentence-final `.` → `।`, and punctuation (`. ! ? , ; : । " ( )`) split into separate tokens.
- Kept only pairs with 1–25 tokens per side, a Hindi/English length ratio between 0.4 and 2.5, a Devanagari Hindi side and an English side without Devanagari.
- Vocabulary capped with the Keras `Tokenizer(num_words=…)`; rarer words become `<OOV>`.

The text is lowercased and punctuation is separated for French too. Each model must be given text cleaned **exactly** as it was cleaned for training, which is why `app.py` has a separate cleaning function per language.

## Results

| Model | Validation loss | Test loss | Test perplexity | BLEU (greedy) | BLEU (beam 4) |
|---|---|---|---|---|---|
| English → Hindi | 3.41 | 3.44 | 31.2 | 9.93 | **10.74** |
| English → French | 0.71 | 1.53 | - | - | - |

How to read these numbers:

- **Loss** is the average cross-entropy per word on sentences the model never trained on (lower is better); `exp(loss)` is the perplexity.
- **BLEU** (0–100, measured on 300 held-out test sentences) compares the output with a single human translation. It is strict and under-rates correct translations that are worded differently, so it is best used to compare model versions.
- The Hindi test set is the same noisy mix of long web and government text as the training data, so these scores say little about short everyday sentences, which the model handles much better (see the examples above).
- Beam search beats greedy decoding by about 0.8 BLEU and fixes several greedy mistakes.

## Known limitations

- **Short, simple, correctly spelled sentences work best.** Input is limited to 15 (French) or 25 (Hindi) words; the rest is cut off.
- Words that were not in the training vocabulary are treated as unknown.
- **Hindi:** formal and informal "you" (आप / तुम) and verb agreement are sometimes wrong (for example "Where do you live?" → "तुम कहाँ रहते हैं?"). Some sentences repeat a word ("I will come next day" → "मैं अगले दिन मैं आऊँगा") or use the wrong construction ("I am hungry" → "मैं भूख लगी हूँ"). The training data is mostly long, noisy text with few conversational sentences.
- **French:** short phrases without a verb can be mistranslated ("Near the sea" → "Ferme la mer."), and some words can be dropped ("Tom is my brother" → "Mon frère.").
- This is a small model trained from scratch. It is a learning and portfolio project, not a replacement for a professional translator.

## Project structure

```
.
├── app.py                     # Streamlit app (models, decoding and UI)
├── requirements.txt
├── README.md
├── notebooks/
│   ├── English_to_French.ipynb
│   └── English_to_Hindi.ipynb
└── saved_model/
|   ├── fr/
|   │   ├── config.json
|   │   ├── eng_tokenizer.json
|   │   ├── fr_tokenizer.json
|   │   ├── encoder.weights.h5
|   │   └── decoder.weights.h5
└── saved_model_hindi/
    ├── hi/
    |   ├── config.json
    |   ├── eng_tokenizer.json
    |   ├── hin_tokenizer.json
    |   ├── encoder.weights.h5
    |   └── decoder.weights.h5
```

Keras 3 only reads weight files whose names end in `.weights.h5`. If yours are named `encoder_weights.h5` / `decoder_weights.h5`, the app makes correctly named copies on first load.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

> **Use the same TensorFlow/Keras version that trained the models.** Weights saved with Keras 3 (`.weights.h5`) need Keras 3 to load. In Colab, run `print(tf.__version__)` and pin that exact version in `requirements.txt`.

## Training and exporting a model

1. Open the notebook in Google Colab and set `CSV_PATH`.
2. Run all cells. Every tunable setting (vocabulary caps, dropout, batch size, patience) is in the first code cell.
3. The last cell saves the weights, tokenizers and `config.json` (which also stores the loss and BLEU scores shown in the app), then reloads them into fresh models and stops with an error if anything differs. When it prints `EXPORT OK`, unzip the download into `saved_model/<language code>/`.

## Adding another language

1. Train a model with the notebook and export it into `saved_model/<code>/`.
2. Add an entry to the `LANGUAGES` dictionary in `app.py` with its folder, a cleaning function that matches the training cleaning, a detokenizer and a few example sentences.

## Deployment notes

The weight files are large (the French decoder is 93 MB and the Hindi decoder 72 MB, about 225 MB for everything). GitHub rejects files over 100 MB, so use **Git LFS** for `*.h5` or host the weights on Hugging Face Hub if the vocabulary grows.

## Ideas for improvement

- More and cleaner conversational data (for example Tatoeba, or the short sentences of the IIT Bombay corpus).
- Subword tokenisation (SentencePiece / BPE) instead of whole words.
- Evaluate the French model with the same BLEU and test-loss cells.
- A Transformer model for comparison.

## Tech stack

Python · TensorFlow / Keras · Streamlit · pandas · scikit-learn · NLTK (BLEU)

## Author

Developed by **Karan Gojiya** · [GitHub](https://github.com/KaranGojiya) · [LinkedIn](https://www.linkedin.com/in/karan-gojiya)
