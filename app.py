import re
import string
import pickle
import numpy as np
import gradio as gr
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---- Config (must match training) ----
MAXLEN = 300
MODEL_PATH = "imdb_bilstm_best.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# ---- Load model + tokenizer ----
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# ---- Preprocessing (identical to training) ----
NEGATION_WORDS = {"not", "no", "nor", "never", "none", "neither", "n't"}
STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS

CONTRACTIONS_MAP = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "can't": "cannot",
    "couldn't": "could not", "shouldn't": "should not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "i'm": "i am", "you're": "you are",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "they're": "they are", "i've": "i have",
    "you've": "you have", "we've": "we have", "they've": "they have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "i'd": "i would", "you'd": "you would", "let's": "let us",
}


def fix_contractions(text: str) -> str:
    for contraction, expansion in CONTRACTIONS_MAP.items():
        text = re.sub(r"\b" + contraction + r"\b", expansion, text, flags=re.IGNORECASE)
    return text


def preprocess_text(text: str) -> str:
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = fix_contractions(text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)


def predict_sentiment(review_text: str):
    if not review_text or not review_text.strip():
        return {"Positive": 0.5, "Negative": 0.5}

    cleaned = preprocess_text(review_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAXLEN, padding="post", truncating="post")

    prob_positive = float(model.predict(padded, verbose=0)[0][0])
    prob_negative = 1 - prob_positive

    return {"Positive": prob_positive, "Negative": prob_negative}


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=6,
        placeholder="Type or paste a movie review here...",
        label="Movie Review",
    ),
    outputs=gr.Label(num_top_classes=2, label="Sentiment"),
    title="IMDB Sentiment Classifier",
    description=(
        "A Bidirectional LSTM with pretrained GloVe embeddings, trained on the "
        "IMDB 50K movie review dataset. Enter a review to predict sentiment."
    ),
    examples=[
        ["This movie was an absolute masterpiece, the acting and story blew me away."],
        ["Total waste of time. The plot made no sense and the acting was terrible."],
        ["It wasn't as bad as I expected, but it definitely wasn't good either."],
    ],
)

if __name__ == "__main__":
    demo.launch()