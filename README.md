---

# **Twitter Sentiment Classification & Error Analysis**

This repository contains an end-to-end sentiment analysis pipeline built using the **Sentiment140** Twitter dataset.
The project follows industry-standard machine learning procedures such as preprocessing, multi-model evaluation, and detailed error analysis.

Originally created as an academic assignment, the project has been extended into a research-style implementation with structured methodology, experiments, and model performance comparisons.

---

## **Project Overview**

This project uses the entire Sentiment140 dataset for training machine learning models.
From the full dataset of 1.6 million tweets, the following splits were created:

Training: all tweets except those reserved for evaluation
Validation: 4,000 tweets
Test (Evaluation): 4,000 tweets
This ensures the model is trained on the maximum amount of available data, while keeping clean and balanced validation and test subsets for proper evaluation.

The project includes:

* Real-world tweet preprocessing
* Multiple embedding methods
* LSTM-based model architectures
* Error analysis on misclassified samples

---

## **Dataset**

This project uses the **Sentiment140** dataset published on Kaggle:

**Dataset link:**
[https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

**Primary file used:**
`training.1600000.processed.noemoticon.csv`

> Note: Due to GitHub file-size limitations, the dataset is **not included** in this repository.
> Please download it from Kaggle and place it in:
> `data/train/` before running the notebook.

---

## **Repository Structure**

```
├── data/                       # Place training/validation/test data here (not included)
├── notebook/
│   └── sentiment_analysis_source_code.ipynb
├── models/                     # Saved models and embeddings
├── results/                    # Metrics tables, plots, error samples
├── README.md                   # Project documentation
└── requirements.txt
```

---

# **1. Data Cleaning & Preprocessing**

Tweets undergo the following preprocessing steps:

* Lowercasing
* URL removal
* Removal of @mentions and hashtags
* Punctuation removal
* Tokenization
* OOV handling using BPE/WordPiece or FastText-style n-grams

Example comparison (10 samples shown in results folder):

| Raw Tweet                                                | Cleaned Tweet      |
| -------------------------------------------------------- | ------------------ |
| "OMG I LOVE this!! [https://t.co/xyz](https://t.co/xyz)" | "omg i love this"  |
| "@user this is NOT good!! #fail"                         | "this is not good" |

---

# **2. Multi-Representation + Multi-Model Evaluation**

The project evaluates **nine total models** using:

### **Text Representations**

1. TF–IDF
2. Word2Vec (Skip-Gram)
3. GloVe (Pretrained Embeddings)

### **Sequential Deep Learning Models**

1. LSTM
2. GRU
3. BiLSTM

Each embedding × model combination is trained and evaluated.

---

## **Benchmark Table (Example Format)**

| Embedding | Model  | Accuracy | Precision | Recall | F1 Score |
| --------- | ------ | -------- | --------- | ------ | -------- |
| TF-IDF    | LSTM   | ...      | ...       | ...    | ...      |
| TF-IDF    | GRU    | ...      | ...       | ...    | ...      |
| Word2Vec  | GRU    | ...      | ...       | ...    | ...      |
| GloVe     | BiLSTM | ...      | ...       | ...    | ...      |

**Minimum required performance:**

* F1 ≥ 85%
* Additional bonus for higher F1 tiers (90%, 93%, 96%)

All experiments avoid data leakage and maintain a balanced test set.

---

# **3. Industry-Aligned Error Analysis**

A detailed analysis was performed on **50 misclassified tweets** from the best-performing model.
Errors were categorized into:

### **1. Sarcasm**

Example:
"Great, another Monday. Exactly what I needed."
Issue: No explicit negative words; sarcasm misinterpreted as positive.

### **2. Negation Errors**

Example:
"I am not impressed with this update."
Issue: Misinterpreting negation structure.

### **3. OOV Words / Slang**

Example:
"This phone is lit"
Issue: Informal slang not captured in embeddings.

### **4. Multi-Topic Tweets**

Example:
"Camera quality is bad but battery is excellent."
Issue: Conflicting sentiments within a single tweet.

### **5. Domain Drift**

Example:
"Fed raised rates again today"
Issue: Domain-specific financial terminology not seen during training.

Each category includes explanations and recommended model improvements.

---

# **Key Insights**

* Traditional embeddings (TF-IDF, Word2Vec, GloVe) remain effective for tweet-length text.
* BiLSTM with GloVe embeddings achieved the strongest performance.
* Sarcasm, negation, and slang remain major challenges.
* Error analysis highlights several avenues for model improvement.

---

# **Technologies Used**

* Python, NumPy, Pandas
* Scikit-Learn
* TensorFlow / Keras
* Gensim (Word2Vec)
* Pretrained GloVe Embeddings
* NLTK / spaCy
* Matplotlib / Seaborn

---

# **How to Run**

```bash
git clone <your-repo-url>
cd twitter-sentiment-analysis
pip install -r requirements.txt
jupyter notebook
```

Open `sentiment_analysis_source_code.ipynb` and run all cells in order.

---

# **Future Work**

* Incorporating Transformer-based models (BERT, RoBERTa)
* Emoji-aware sentiment representations
* Sarcasm detection modules
* Multi-aspect sentiment analysis
* Domain-adaptive model training

---

# **Contributions**

Contributions are welcome.
Please open an issue for improvements or bug reports.

---

# **License**

This project is open-source and available for educational and research use.
