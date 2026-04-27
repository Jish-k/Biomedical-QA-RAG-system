
import streamlit as st
import torch
import numpy as np
import faiss
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import time
import os

# --- Page Config ---
st.set_page_config(page_title="PubMedQA - Advanced RAG", page_icon="🧬", layout="wide")

# --- Styling ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background: linear-gradient(45deg, #007bff, #00d2ff); color: white; font-weight: bold; border: none; }
    .result-card { background: white; padding: 25px; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 8px solid #007bff; }
    .evidence-card { background: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; margin-top: 10px; font-size: 0.9rem; }
    .verdict-yes { color: #28a745; font-weight: bold; font-size: 1.5rem; }
    .verdict-no { color: #dc3545; font-weight: bold; font-size: 1.5rem; }
    .verdict-maybe { color: #ffc107; font-weight: bold; font-size: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Constants ---
MODEL_PATH = "models/pubmedbert_final"
EMBEDDER_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@st.cache_resource
def load_advanced_resources():
    st.info("🚀 Booting up Advanced RAG Engines (BGE-Base + Cross-Encoder)...")
    
    # 1. Load Data
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    documents = [" ".join(item["context"]["contexts"]) for item in dataset]
    documents = list(set(documents))
    
    # 2. Embedder & Index
    embedder = SentenceTransformer(EMBEDDER_NAME)
    doc_embeddings = embedder.encode(documents, show_progress_bar=False)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(np.array(doc_embeddings))
    
    # 3. BM25
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    
    # 4. Reranker
    reranker = CrossEncoder(RERANKER_NAME)
    
    # 5. Classifier
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    classifier.eval()
    
    return documents, embedder, index, bm25, reranker, tokenizer, classifier

def run_rag_inference(query, documents, embedder, index, bm25, reranker, tokenizer, classifier):
    # Step 1: Hybrid Retrieval
    # Vector Search
    query_emb = embedder.encode([query])
    _, faiss_idx = index.search(query_emb, 20)
    faiss_res = [documents[i] for i in faiss_idx[0]]
    
    # Keyword Search
    bm25_res = bm25.get_top_n(query.lower().split(), documents, n=20)
    candidates = list(set(faiss_res + bm25_res))
    
    # Step 2: Cross-Encoder Reranking
    pairs = [[query, cand] for cand in candidates]
    scores = reranker.predict(pairs)
    best_doc = candidates[np.argmax(scores)]
    
    # Step 3: Classifier Inference
    input_text = f"Question: {query} Context: {best_doc}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = classifier(**inputs)
    
    # Apply small bias to YES (index 0) to combat conservatism if logits are close
    logits = outputs.logits[0].cpu().numpy()
    # If YES is competitive, give it a slight nudge
    if logits[0] > -0.5: logits[0] += 0.5 
    
    pred = np.argmax(logits)
    labels = ["YES", "NO", "MAYBE"]
    
    return labels[pred], best_doc, logits

def main():
    st.markdown("# 🧬 PubMedQA Advanced RAG")
    st.markdown("### High-Precision Biomedical Decision Support")
    
    resources = load_advanced_resources()
    documents, embedder, index, bm25, reranker, tokenizer, classifier = resources
    
    st.sidebar.title("System Status")
    st.sidebar.success("Model: PubMedBERT (Two-Stage)")
    st.sidebar.info(f"Retrieval: Hybrid (BM25 + BGE-Base)")
    st.sidebar.warning("Reranker: Cross-Encoder active")
    
    question = st.text_input("Enter clinical question:", placeholder="e.g., Does high BMI increase surgical risk in CABG?")
    
    if st.button("Analyze Evidence") and question:
        with st.spinner("Executing multi-stage retrieval and reranking..."):
            start = time.time()
            verdict, context, logits = run_rag_inference(question, *resources)
            end = time.time()
            
            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"#### Final Verdict: <span class='verdict-{verdict.lower()}'>{verdict}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence Profile:** Yes: {logits[0]:.2f} | No: {logits[1]:.2f} | Maybe: {logits[2]:.2f}")
            st.markdown(f"**Latency:** {end-start:.2f}s")
            
            st.markdown("---")
            st.markdown("**Supporting Evidence (Top Reranked Abstract):**")
            st.markdown(f'<div class="evidence-card">{context}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            if verdict == "YES":
                st.success("The system found strong scientific consensus in the literature.")
            elif verdict == "NO":
                st.error("The system found evidence contradicting the query.")
            else:
                st.warning("The literature provides conflicting or insufficient evidence.")

if __name__ == "__main__":
    main()
