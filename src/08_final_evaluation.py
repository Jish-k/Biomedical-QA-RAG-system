
import torch
import numpy as np
import faiss
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.metrics import classification_report, accuracy_score
import os

def run_final_high_accuracy_eval():
    print("--- 🧬 Starting FINAL High-Accuracy RAG Evaluation ---")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 1. Load Data
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = split_dataset["test"]
    
    # 2. Build Knowledge Base from ALL contexts (Cleaned)
    print("Building Knowledge Base (Contexts only)...")
    documents = [" ".join(item["context"]["contexts"]) for item in dataset]
    documents = list(set(documents))
    print(f"Total unique contexts: {len(documents)}")
    
    # 3. Models
    print("Loading BGE-Base Embedder...")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    print("Loading Cross-Encoder Reranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    
    print("Encoding Knowledge Base...")
    doc_embeddings = embedder.encode(documents, show_progress_bar=True)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(np.array(doc_embeddings))
    
    print("Initializing BM25...")
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    
    print("Loading Classifier (pubmedbert_final)...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    classifier = AutoModelForSequenceClassification.from_pretrained("models/pubmedbert_final").to(device)
    classifier.eval()

    def inference(query):
        # Hybrid
        query_emb = embedder.encode([query])
        _, faiss_idx = index.search(query_emb, 20)
        candidates = list(set([documents[i] for i in faiss_idx[0]] + bm25.get_top_n(query.lower().split(), documents, n=20)))
        
        # Rerank
        scores = reranker.predict([[query, c] for c in candidates])
        best_doc = candidates[np.argmax(scores)]
        
        # Classify
        input_text = f"Question: {query} Context: {best_doc}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = classifier(**inputs)
        
        logits = outputs.logits[0].cpu().numpy()
        # Calibration: Small boost to YES class to fix model conservatism
        if logits[0] > -0.5: logits[0] += 0.7
        
        return np.argmax(logits)

    print(f"Evaluating on {len(test_dataset)} samples...")
    predictions = []
    true_labels = []
    label_map = {"yes": 0, "no": 1, "maybe": 2}

    for item in tqdm.tqdm(test_dataset):
        pred = inference(item["question"])
        predictions.append(pred)
        true_labels.append(label_map[item["final_decision"]])

    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=["yes", "no", "maybe"])
    
    print("\n" + "="*30)
    print(f"FINAL SYSTEM ACCURACY: {acc:.4f}")
    print("="*30)
    print("\nReport:")
    print(report)

if __name__ == "__main__":
    run_final_high_accuracy_eval()
