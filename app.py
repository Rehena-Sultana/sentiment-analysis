import os
import re
import string
import pickle
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk

# ---- Page Configuration ----
st.set_page_config(
    page_title="IMDB Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
)

# ---- NLTK Downloads ----
@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

setup_nltk()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---- Model & Config ----
MAXLEN = 300
MODEL_PATH = "imdb_bilstm_best.h5"
TOKENIZER_PATH = "tokenizer.pkl"

@st.cache_resource
def load_sentiment_model():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_sentiment_model()

# ---- Preprocessing ----
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
        return 0.5, 0.5, ""

    cleaned = preprocess_text(review_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAXLEN, padding="post", truncating="post")

    prob_positive = float(model.predict(padded, verbose=0)[0][0])
    prob_negative = 1.0 - prob_positive
    return prob_positive, prob_negative, cleaned

# ---- UI Header ----
st.title("🎬 IMDB Sentiment Classifier")
st.caption("Powered by Bidirectional LSTM + Pretrained GloVe Word Embeddings")

st.markdown(
    "Enter a movie review below or select one of the sample reviews to predict whether "
    "the sentiment is **Positive** or **Negative**."
)

# Callbacks for sample buttons
def set_review(text: str):
    st.session_state["review_input"] = text

def clear_review():
    st.session_state["review_input"] = ""

examples = [
    ("🌟 Positive Example", "This movie was an absolute masterpiece, the acting and story blew me away."),
    ("👎 Negative Example", "Total waste of time. The plot made no sense and the acting was terrible."),
]

st.write("**Try an example:**")
cols = st.columns(2)
for i, (label, text) in enumerate(examples):
    cols[i].button(label, key=f"ex_btn_{i}", on_click=set_review, args=(text,), use_container_width=True)


if "review_input" not in st.session_state:
    st.session_state["review_input"] = ""

review = st.text_area(
    label="Movie Review",
    key="review_input",
    height=150,
    placeholder="Type or paste a movie review here...",
)

col_btn1, col_btn2, _ = st.columns([1.5, 1, 3.5])
with col_btn1:
    submit = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
with col_btn2:
    st.button("Clear", on_click=clear_review, use_container_width=True)

if submit and review.strip():
    with st.spinner("Analyzing review..."):
        prob_pos, prob_neg, cleaned_text = predict_sentiment(review)

    st.divider()

    if prob_pos >= 0.5:
        st.success(f"### 🎉 Positive Sentiment ({prob_pos * 100:.1f}% Confidence)")
    else:
        st.error(f"### 👎 Negative Sentiment ({prob_neg * 100:.1f}% Confidence)")

    col_pos, col_neg = st.columns(2)
    col_pos.metric("Positive Probability", f"{prob_pos * 100:.2f}%")
    col_neg.metric("Negative Probability", f"{prob_neg * 100:.2f}%")

    st.progress(prob_pos, text=f"Positive: {prob_pos * 100:.1f}% | Negative: {prob_neg * 100:.1f}%")

    with st.expander("🛠️ View Cleaned Text (NLP Preprocessing pipeline)"):
        st.code(cleaned_text if cleaned_text else "(empty)", language="text")

elif submit:
    st.warning("Please enter or select a review before analyzing.")