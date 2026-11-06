# 🎬 IMDB Movie Review Sentiment Analysis

[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://sentiment-analysis-bnwu.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

An end-to-end Deep Learning Natural Language Processing (NLP) web application that classifies movie reviews as **Positive** or **Negative**. Built with **TensorFlow/Keras** and **Streamlit**, the classifier uses a stacked **Bidirectional LSTM (BiLSTM)** architecture with pretrained **GloVe (840B, 300d)** word embeddings trained on the **IMDB 50K Movie Reviews Dataset**.

🔗 **Live Demo:** [IMDB Movie Review Sentiment Analysis](https://sentiment-analysis-bnwu.onrender.com/)

---

## ✨ Features

- **Real-Time Sentiment Prediction:** Classifies any movie review into **Positive** or **Negative** sentiment along with exact probability scores and a visual confidence bar.
- **Negation-Aware NLP Pipeline:** Cleans HTML tags, URLs, smart quotes, punctuation, and numbers, expands English contractions (`"wasn't"` -> `"was not"`), and filters out stopwords while **preserving negation words** (`not`, `no`, `never`, `neither`, etc.) so negative phrases retain their true meaning.
- **Interactive Sample Reviews:** One-click positive and negative example buttons for instant testing.
- **Transparent Preprocessing View:** Includes an expandable inspector in the UI to view the exact cleaned and tokenized text fed into the BiLSTM model.

---

## 🧠 Model Architecture & Training

The model was trained and evaluated in [`sentiment_analysis_imdb_dataset.ipynb`](sentiment_analysis_imdb_dataset.ipynb) on the [Kaggle IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

### Dataset & Tokenization
- **Dataset Split:** 70% Train / 15% Validation / 15% Test (stratified by sentiment label)
- **Vocabulary Size (`NUM_WORDS`):** `20,000` (with `<OOV>` token for out-of-vocabulary words, fitted strictly on the training set to prevent data leakage)
- **Max Sequence Length (`MAXLEN`):** `300` tokens (`post` padding and `post` truncation)
- **Pretrained Word Embeddings:** **GloVe 840B 300d** (`300`-dimensional vectors covering >95% of the vocabulary)

### BiLSTM Network Architecture

| Layer | Type | Configuration |
| :--- | :--- | :--- |
| **1. Embedding** | `Embedding` | `input_dim=20000`, `output_dim=300`, initialized with GloVe 300d matrix |
| **2. Spatial Dropout** | `SpatialDropout1D` | `rate=0.3` |
| **3. BiLSTM Layer 1** | `Bidirectional(LSTM)` | `128` units (`256` bidirectional), `return_sequences=True`, `recurrent_dropout=0.2` |
| **4. BiLSTM Layer 2** | `Bidirectional(LSTM)` | `64` units (`128` bidirectional), `recurrent_dropout=0.2` |
| **5. Fully Connected** | `Dense` | `64` units, `activation="relu"` |
| **6. Dropout** | `Dropout` | `rate=0.5` |
| **7. Output** | `Dense` | `1` unit, `activation="sigmoid"` |

- **Optimizer:** Adam (`learning_rate=1e-3`)
- **Loss Function:** Binary Crossentropy
- **Callbacks:** `EarlyStopping` (`patience=4`), `ReduceLROnPlateau` (`factor=0.5`, `patience=2`), and `ModelCheckpoint`
- **Performance:** Achieved **~89.5% – 90.5% accuracy** on the held-out validation/test sets across notebook iterations.

---

## 🛠️ NLP Preprocessing Pipeline

Every input review passes through the following preprocessing steps before inference:
1. **HTML & URL Removal:** Strips HTML tags (`<br />`, etc.) using `BeautifulSoup` and removes URLs via regex.
2. **Unicode & Contraction Normalization:** Normalizes unicode quotes/dashes to ASCII and expands contractions (e.g., `"don't"` -> `"do not"`, `"can't"` -> `"cannot"`).
3. **Text Cleaning:** Converts text to lowercase, removes punctuation and standalone numbers, and collapses extra whitespace.
4. **Negation-Safe Stopword Removal:** Tokenizes text with NLTK's `word_tokenize` and removes standard English stopwords **except** negation terms (`not`, `no`, `nor`, `never`, `none`, `neither`, `n't`).
5. **Sequence Encoding & Padding:** Converts cleaned text into integer sequences via `tokenizer.pkl` and pads/truncates to length `300`.

---

## 📁 Project Structure

```text
sentiment-analysis/
├── app.py                                  # Streamlit web application
├── imdb_bilstm_best.h5                     # Trained Keras BiLSTM model weights
├── tokenizer.pkl                           # Pickled Keras Tokenizer fitted on training data
├── sentiment_analysis_imdb_dataset.ipynb   # Jupyter Notebook for data preprocessing & model training
├── requirements.txt                        # Python dependencies
├── .python-version                         # Python runtime version (3.11.9) for Render
└── README.md                               # Project documentation
```

---

## 🚀 Getting Started Locally

### Prerequisites
- Python `3.10` or `3.11` (configured with `3.11.9` in `.python-version`)
- `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/Rehena-Sultana/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv myenv
.\myenv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Render

This application is deployed as a **Web Service** on [Render](https://render.com/):
- **Live URL:** [https://sentiment-analysis-bnwu.onrender.com/](https://sentiment-analysis-bnwu.onrender.com/)
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- **Platform Optimization:** `requirements.txt` automatically installs the lightweight `tensorflow-cpu` package on Linux servers (`sys_platform == 'linux'`) to optimize memory usage and build times on Render.
