#!/usr/bin/env python
# coding: utf-8

# # 06 — PubMedBERT Two-Stage Training (Pretrain + Fine-tune)
# 
# This notebook implements a two-stage training strategy for PubMedQA:
# 1. **Pretraining** on the large artificial dataset (`pqa_artificial`, 211k rows).
# 2. **Fine-tuning** on the expert-labeled dataset (`pqa_labeled`, 1k rows).
# 
# This is the standard approach to maximize the utility of noisy but massive artificial labels.

# ## 1. Import Libraries

# In[ ]:


import os
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn import CrossEntropyLoss
import seaborn as sns
import matplotlib.pyplot as plt


# ## 2. Hardware Setup

# In[ ]:


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ## 3. Load Datasets

# In[ ]:


print("Loading pqa_artificial (211k rows) for pretraining...")
dataset_artificial = load_dataset("pubmed_qa", "pqa_artificial")
# Limit to 2500 rows for 15-minute deadline
dataset_artificial["train"] = dataset_artificial["train"].select(range(2500))

print("Loading pqa_labeled (1k rows) for fine-tuning...")
dataset_labeled = load_dataset("pubmed_qa", "pqa_labeled")

print(f"Artificial rows: {len(dataset_artificial['train'])}")
print(f"Labeled rows: {len(dataset_labeled['train'])}")


# ## 4. Preprocessing

# In[ ]:


model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

label_map = {"yes": 0, "no": 1, "maybe": 2}

def preprocess_function(example):
    # Combine question and context
    context = " ".join(example["context"]["contexts"])
    example["input_text"] = f"Question: {example['question']} Context: {context}"

    # Encode label
    example["label"] = label_map[example["final_decision"]]
    return example

def tokenize_function(examples):
    return tokenizer(
        examples["input_text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

print("Preprocessing artificial dataset...")
dataset_artificial = dataset_artificial.map(preprocess_function)
dataset_artificial = dataset_artificial.map(tokenize_function, batched=True)

print("Preprocessing labeled dataset...")
dataset_labeled = dataset_labeled.map(preprocess_function)
dataset_labeled = dataset_labeled.map(tokenize_function, batched=True)

dataset_artificial.set_format("torch", columns=["input_ids", "attention_mask", "label"])
dataset_labeled.set_format("torch", columns=["input_ids", "attention_mask", "label"])


# ## 5. Train/Test Split for Labeled Data

# In[ ]:


# Split labeled data into 80% train, 20% test
labeled_split = dataset_labeled["train"].train_test_split(test_size=0.2, seed=42)
dataset_labeled = DatasetDict({
    "train": labeled_split["train"],
    "test": labeled_split["test"]
})
print(dataset_labeled)


# ## 6. Define Evaluation Metrics

# In[ ]:


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# ## 7. Stage 1: Pretraining on Artificial Data

# In[ ]:


pretrain_args = TrainingArguments(
    output_dir="./results/pubmedbert_pretraining",
    num_train_epochs=1,  # 1 epoch on 211k rows is already substantial
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=500,
    save_strategy="no", # Don't save intermediate checkpoints to save space
    report_to="none"
)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

pretrain_trainer = Trainer(
    model=model,
    args=pretrain_args,
    train_dataset=dataset_artificial["train"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting Pretraining...")
pretrain_trainer.train()


# ## 8. Save Pretrained Model

# In[ ]:


os.makedirs("models/pubmedbert_pretrained", exist_ok=True)
pretrain_trainer.save_model("models/pubmedbert_pretrained")
print("Pretrained model saved to models/pubmedbert_pretrained")


# ## 9. Stage 2: Fine-tuning on Labeled Data

# In[ ]:


finetune_args = TrainingArguments(
    output_dir="./results/pubmedbert_finetuning",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

finetune_trainer = Trainer(
    model=model,  # Uses the model already loaded and pretrained
    args=finetune_args,
    train_dataset=dataset_labeled["train"],
    eval_dataset=dataset_labeled["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting Fine-tuning...")
finetune_trainer.train()


# ## 10. Final Evaluation

# In[ ]:


results = finetune_trainer.evaluate()
print("Final Results:", results)


# ## 11. Save Final Model

# In[ ]:


os.makedirs("models/pubmedbert_final", exist_ok=True)
finetune_trainer.save_model("models/pubmedbert_final")
print("Final model saved to models/pubmedbert_final")

