# 🏥 Clinical Text Summarization and Classification

This project focuses on processing clinical transcriptions for two core tasks:  
1. **Abstractive Summarization** of clinical notes  
2. **Classification** of transcriptions into medical specialties  

---

## 🔍 Problem Statement

Medical transcriptions contain verbose clinical narratives. The aim is to:
- Summarize these transcriptions to concise and factual versions
- Classify each transcription into its corresponding medical specialty

---

## 📁 Dataset

- Sourced from **MT Samples**
- Columns include: `description`, `medical_specialty`, `sample_name`, `transcription`, and `keywords`
- Contains around **5000 samples**

---

## 🧠 Task 1: Clinical Text Classification

### ✅ Models Used
- **Naive Bayes** – F1-score: 76.39%
- **SVM** – F1-score: 80.21% *(Best)*
- **Decision Tree** – F1-score: 69.96%
- **Random Forest** – F1-score: 74.53%
- **XGBoost** – F1-score: 76.48%

### 🧪 Evaluation Metrics
- Accuracy, Precision, Recall, F1-score

---

## 📝 Task 2: Clinical Text Summarization

### ✅ Models Used
- **BioGPT**
- **BART**
- **Pegasus**
- **T5**

### 📊 Evaluation Metrics
- **ROUGE-1, ROUGE-2, ROUGE-L**
- **BERTScore** (Precision, Recall, F1)

---

## ⚙️ Preprocessing & Feature Engineering

- Tokenization, Stopword Removal, Lemmatization, Stemming
- Vectorization: **TF-IDF**, **Bag of Words**
- Embeddings: **FastText**, **GloVe**, **Word2Vec**
- Domain-specific: **ClinicalBERT**, **BioBERT**, **SciBERT**
- NER: **Med7**, **MedicalNER**

---

## 💻 Tools & Libraries

- Python, scikit-learn, XGBoost, Transformers (Hugging Face)
- Matplotlib, Seaborn, Pandas, Numpy
- BERTScore, ROUGE, nltk, spaCy

