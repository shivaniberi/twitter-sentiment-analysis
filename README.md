
# **Twitter Sentiment Classification & Error Analysis**

This repository contains an end-to-end sentiment analysis pipeline built using the **Sentiment140** Twitter dataset.
The project follows industry-standard machine learning workflows, including preprocessing, baseline modeling, transformer-based modeling, and detailed error analysis.

Originally developed as an academic assignment, the project has been expanded into a research-oriented implementation with structured methodology, evaluation, and insights into model failures.

---

## **Project Overview**

This project uses the **entire Sentiment140 dataset** for training machine learning models.
From the full dataset of 1.6 million tweets, the following splits were created:

* **Training:** all tweets except evaluation sets
* **Validation:** 4,000 tweets
* **Test (Evaluation):** 4,000 tweets

Using the full dataset ensures maximum exposure to linguistic variations, slang, abbreviations, and informal writing styles commonly found on Twitter.

The project includes:

* Real-world tweet preprocessing
* Baseline classical models and deep learning architectures
* Transition from classical models to transformers due to context limitations
* Error analysis of misclassified samples

---

## **Dataset**

This project uses the **Sentiment140** dataset from Kaggle:

Dataset link:
[https://www.kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

Primary file required:
`training.1600000.processed.noemoticon.csv`

> Due to GitHub file-size limitations, the dataset is **not included** in this repository.
> Download it from Kaggle and place it in:
> `data/train/`

---

## **Repository Structure**

```
├── data/                       # Place training/validation/test data here (not included)
├── notebook/
│   └── sentiment_analysis_source_code.ipynb
├── models/                     # Saved models
├── results/                    # Metrics tables, plots, error samples
└── README.md
```

---

# **1. Data Cleaning & Preprocessing**

Tweets are cleaned using real-world moderation-style processing:

* Lowercasing
* URL removal
* Removal of @mentions and hashtags
* Punctuation removal
* Tokenization
* Handling out-of-vocabulary terms

Example comparison (full table in results folder):

| Raw Tweet                        | Cleaned Tweet      |
| -------------------------------- | ------------------ |
| “OMG I LOVE this!!”              | “omg i love this”  |
| “@user this is NOT good!! #fail” | “this is not good” |

---

# **2. Baseline Models and Motivation for Transformers**

Initially, several classical and deep learning models were explored, including:

* TF–IDF based classifiers
* Word2Vec and GloVe embeddings
* LSTM, GRU, and BiLSTM architectures

While these models performed reasonably well, they showed **significant limitations**, especially in:

* Capturing long-range dependencies
* Understanding sarcasm
* Handling multi-topic tweets
* Dealing with informal language, slang, and abbreviations
* Managing contextual polarity shifts (e.g., negations)

Because of these challenges, the project progressed to **transformer-based models**, which provided substantial improvements in contextual understanding and overall performance.

---

# **3. Error Analysis**

A detailed analysis of **50 misclassified tweets** from the best-performing model revealed key problem categories:

### **1. Sarcasm**

Example:
“Great, another Monday. Exactly what I needed.”
Issue: No explicit negative tokens; sarcasm is implicit.

### **2. Negation Handling**

Example:
“I am not impressed with this update.”
Issue: Basic models fail to invert sentiment correctly.

### **3. OOV Words / Slang**

Example:
“This phone is lit”
Issue: Informal slang not captured by classical embeddings.

### **4. Multi-Topic Tweets**

Example:
“Camera quality is bad but battery is excellent.”
Issue: Mixed polarity within a single text.

### **5. Domain Drift**

Example:
“Fed raised rates again today”
Issue: Financial terminology not seen during training.

The detailed analysis provides insights into why these errors occur and how modern NLP architectures address them.

---

# **Key Insights**

* Classical models struggle with contextual interpretation, slang, sarcasm, and negation.
* Transformer architectures significantly improve contextual understanding.
* Preprocessing plays a critical role in stabilizing model performance.
* Error analysis is essential for identifying weaknesses and guiding future work.

---

# **How to Run the Project**

```bash
git clone <your-repo-url>
cd twitter-sentiment-analysis
jupyter notebook
```

Open and execute `sentiment_analysis_source_code.ipynb`.

---

# **Future Work**

* Full fine-tuning of transformer models (BERT, RoBERTa, DistilBERT)
* Emoji-aware sentiment embeddings
* Sarcasm detection modules
* Multi-aspect sentiment classification
* Domain adaptation for evolving Twitter language

---

# **Contributions**

Contributions are welcome.
Please open an issue for discussions, improvements, or feature additions.

---

# **License**

This project is open-source and available for academic and research use.
