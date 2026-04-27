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

def run_advanced_evaluation():
    print("--- 🧬 Starting Advanced RAG Pipeline Evaluation ---")
    
    # 1. Load Data
    print("Loading labeled dataset...")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
    # Create the same split as in the notebook
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_docs_data = split_dataset["train"]
    test_dataset = split_dataset["test"]
    
    # 2. Build Knowledge Base from FULL labeled dataset
    print("Building Knowledge Base from all 1,000 samples...")
    documents = []
    for item in dataset: # Use the full 'train' split which contains all 1,000 samples
        context = " ".join(item["context"]["contexts"])
        long_answer = item["long_answer"]
        documents.append(context)
        documents.append(long_answer)

    documents = list(set(documents)) # Remove duplicates
    print(f"Total unique documents in KB: {len(documents)}")
    
    # 3. Initialize Retrieval Models
    print("Initializing Embedder (BGE) and Reranker (Cross-Encoder)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    
    # Pre-calculate embeddings for FAISS
    print("Encoding documents for Vector Search...")
    doc_embeddings = embedder.encode(documents, show_progress_bar=True)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(np.array(doc_embeddings))
    
    # Initialize BM25 for Keyword Search
    print("Initializing BM25...")
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    
    # 4. Hybrid Retrieval + Reranking Function
    def advanced_retrieve(query, top_k_initial=20, top_k_final=3):
        # Vector Search
        query_embedding = embedder.encode([query])
        _, faiss_indices = index.search(query_embedding, top_k_initial)
        faiss_results = [documents[i] for i in faiss_indices[0]]
        
        # Keyword Search
        tokenized_query = query.lower().split()
        bm25_results = bm25.get_top_n(tokenized_query, documents, n=top_k_initial)
        
        # Combine and Deduplicate
        combined_candidates = list(set(faiss_results + bm25_results))
        
        # Rerank with Cross-Encoder
        pairs = [[query, doc] for doc in combined_candidates]
        scores = reranker.predict(pairs)
        
        # Sort by score
        reranked_indices = np.argsort(scores)[::-1][:top_k_final]
        return [combined_candidates[i] for i in reranked_indices]

    # 5. Load the Two-Stage Trained Model
    model_path = "models/pubmedbert_final"
    print(f"Loading classifier from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    classifier = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    classifier.eval()
    
    # 6. Final QA Logic
    def advanced_answer_question(question):
        # Retrieve top 1 document using hybrid + rerank (matching single-context training)
        docs = advanced_retrieve(question, top_k_final=1)
        
        # Augment input using the EXACT format the model was trained on
        evidence_text = " ".join(docs)
        combined_text = f"Question: {question} Context: {evidence_text}"
        
        inputs = tokenizer(
            combined_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = classifier(**inputs)
        
        pred = torch.argmax(outputs.logits, dim=1).item()
        labels = ["yes", "no", "maybe"]
        return labels[pred]

    # 7. Evaluation
    print(f"Evaluating on {len(test_dataset)} test samples...")
    predictions = []
    true_labels_raw = []
    
    # Mapping string labels to indices if necessary, but here we compare string to string
    # PubMedQA labels are 0, 1, 2 for yes, no, maybe? Let's check.
    # From summary: 0 -> YES, 1 -> NO, 2 -> MAYBE
    for item in tqdm.tqdm(test_dataset):
        pred_label = advanced_answer_question(item["question"])
        predictions.append(pred_label)
        true_labels_raw.append(item["final_decision"])
    
    acc = accuracy_score(true_labels_raw, predictions)
    report = classification_report(true_labels_raw, predictions, target_names=["yes", "no", "maybe"])
    
    print("\n" + "="*30)
    print(f"FINAL ACCURACY: {acc:.4f}")
    print("="*30)
    print("\nClassification Report:")
    print(report)
    
    # Save results to a file
    with open("advanced_rag_results.txt", "w") as f:
        f.write(f"Final Accuracy: {acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
    
    print("\nResults saved to 'advanced_rag_results.txt'")

if __name__ == "__main__":
    run_advanced_evaluation()
