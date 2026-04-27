# PubMedQA: Retrieval-Augmented Generation (RAG) System

This repository contains a Deep Learning project for **Biomedical Question Answering**. The system leverages a RAG pipeline to provide accurate answers to medical queries by retrieving relevant research context before classification.

## 🚀 Project Overview

The core objective is to determine if domain-specific models (PubMedBERT) combined with retrieval techniques outperform standard BERT baselines in the medical domain.

### 🛠️ Pipeline Architecture
1.  **Data Exploration & Preprocessing**: Analyzing the PubMedQA dataset and preparing it for training.
2.  **Domain-Specific Training**: Fine-tuning **PubMedBERT** (BiomedNLP) on the PubMedQA classification task.
3.  **Retrieval System**: Building a vector database using **FAISS** and **BGE Embeddings** to retrieve relevant context.
4.  **Augmentation & Classification**: Combining questions with retrieved evidence to predict answers: **Yes**, **No**, or **Maybe**.

## 📁 Project Structure
- `README.md`: Project title, abstract, and setup instructions.
- `requirements.txt`: Project dependencies.
- `data/`: Sample data and dataset files.
- `notebooks/`: Jupyter Notebooks for data exploration, baseline training, and RAG pipeline evaluation.
- `src/`: Python scripts, Streamlit application (`app.py`), and saved model weights.
- `results/`: Training logs, checkpoints, and evaluation charts.
- `report.pdf`: Final IEEE report detailing methodology and results.

## 💻 How to Run the Demo

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

### 3. Usage
- Enter a medical question in the input box.
- The system will retrieve the top 3 relevant context snippets.
- The fine-tuned model will provide a final decision with a confidence score.

## 📊 Results
- **Model**: PubMedBERT (base-uncased)
- **Accuracy**: ~56% (Final RAG Pipeline)
- **Frameworks**: Transformers, FAISS, PyTorch, Streamlit

---
*Developed as a Major Project for Deep Learning Course (2nd Semester).*
