"""
Fine-tuning script for the PPU intent classifier.

Base model: CAMeL-Lab/bert-base-arabic-camelbert-mix  (Arabic BERT)
    • Pretrained on a large Arabic corpus covering MSA and dialectal Arabic
    • Falls back to bert-base-multilingual-cased if unavailable

Dataset format (JSONL, one example per line) — Arabic primary:
        {"text": "ما هي رسوم التسجيل في قسم هندسة الحاسوب؟", "label": "faq_fees"}
        {"text": "متى يبدأ تسجيل الفصل الثاني؟", "label": "faq_registration"}
        {"text": "هل توجد منح للطلبة المتفوقين؟", "label": "faq_scholarships"}
        {"text": "ما هي متطلبات التخرج لقسم هندسة الحاسوب؟", "label": "faq_graduation"}
        {"text": "أين موقع الجامعة؟", "label": "campus_services"}
        {"text": "السلام عليكم", "label": "greeting"}
        {"text": "What are the CS tuition fees?", "label": "faq_fees"}
        {"text": "When does spring registration start?", "label": "faq_registration"}
        {"text": "This is a random off-topic question", "label": "unknown"}

Notes on data preparation:
    - Keep labels aligned with nlp_engine/intent/labels.json (unknown labels are skipped).
    - Start from official FAQs, handbook PDFs, department pages, and real chat logs.
    - Normalize spelling variants and include both Arabic and English phrasing.
    - Balance classes (aim for 50-200+ examples per intent) to reduce bias.
    - Deduplicate near-identical questions and remove template-like repeats.
    - Include an "unknown" intent with off-topic or out-of-scope queries.
    - Save as UTF-8 JSONL with one object per line (no surrounding list).

Usage
-----
Option A: Edit TRAINING_CONFIG below and run without CLI args
        python -m nlp_engine.intent.fine_tune

Option B: Override via CLI
        python -m nlp_engine.intent.fine_tune \
                --data_path data/training/intent_dataset.jsonl \
                --epochs 5 \
                --batch_size 16 \
                --lr 2e-5 \
                --output_dir data/models/intent
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_LABELS_PATH = Path(__file__).parent / "labels.json"

# Edit this block to tune training from code (CLI args override these defaults).
TRAINING_CONFIG = {
    "data_path": "data/training/intent_dataset.jsonl",
    "output_dir": "data/models/intent",
    "base_model": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "epochs": 5,
    "batch_size": 16,
    "lr": 2e-5,
    "val_split": 0.15,
    "max_len": 128,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IntentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def load_dataset(data_path: str, label2id: Dict[str, int]):
    texts, labels = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["label"] not in label2id:
                logger.warning("Skipping unknown label: %s", row["label"])
                continue
            texts.append(row["text"])
            labels.append(label2id[row["label"]])
    return texts, labels


def train(
    data_path: str,
    output_dir: str,
    base_model: str = "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    val_split: float = 0.15,
    max_len: int = 128,
):
    # Load labels
    with open(_LABELS_PATH, "r", encoding="utf-8") as f:
        labels_list = json.load(f)
    label2id = {lbl: i for i, lbl in enumerate(labels_list)}
    id2label = {i: lbl for i, lbl in enumerate(labels_list)}
    num_labels = len(labels_list)

    # Load data
    texts, labels = load_dataset(data_path, label2id)
    logger.info("Loaded %d examples covering %d unique labels", len(texts), len(set(labels)))

    # Train / val split
    use_val = val_split > 0.0 and val_split < 1.0 and len(texts) > 1
    if use_val:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                texts, labels, test_size=val_split, random_state=42, stratify=labels
            )
        except ValueError as exc:
            logger.warning("Stratified split failed (%s). Falling back to random split.", exc)
            X_train, X_val, y_train, y_val = train_test_split(
                texts, labels, test_size=val_split, random_state=42, stratify=None
            )
    else:
        logger.info("Skipping validation split; using all %d examples for training.", len(texts))
        X_train, y_train = texts, labels
        X_val, y_val = [], []

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = IntentDataset(X_train, y_train, tokenizer, max_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = None
    if X_val:
        val_ds = IntentDataset(X_val, y_val, tokenizer, max_len)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)
    model.to(device)

    # ---- Class-weight balancing ----
    # Compute inverse-frequency weights so under-represented intents
    # (e.g. student_id_query) get proportionally higher loss weight.
    label_counts = Counter(y_train)
    logger.info("Training label distribution: %s", dict(sorted(label_counts.items())))
    weights = torch.tensor(
        [1.0 / (label_counts.get(i, 1)) for i in range(num_labels)],
        dtype=torch.float32,
        device=device,
    )
    # Normalise so the average weight stays ~1 (prevents loss scale shift)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    logger.info("Using class-weighted CrossEntropyLoss (max_w=%.2f, min_w=%.2f)",
                weights.max().item(), weights.min().item())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            batch    = {k: v.to(device) for k, v in batch.items()}
            outputs  = model(**batch)
            # Use weighted loss instead of the default HF loss
            loss = loss_fn(outputs.logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)

        # --- Validate ---
        if val_dl is not None:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_dl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    preds = torch.argmax(model(**batch).logits, dim=-1)
                    correct += (preds == batch["labels"]).sum().item()
                    total += len(batch["labels"])
            val_acc = correct / total
            logger.info("Epoch %d/%d — loss: %.4f — val_acc: %.4f", epoch, epochs, avg_loss, val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info("  ✓ Saved best model (val_acc=%.4f) to %s", best_val_acc, output_dir)
        else:
            logger.info("Epoch %d/%d — loss: %.4f", epoch, epochs, avg_loss)

    if val_dl is None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved model to %s", output_dir)
    else:
        logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for PPU intent classification")
    parser.add_argument("--data_path", default=TRAINING_CONFIG["data_path"], help="Path to JSONL training file")
    parser.add_argument("--output_dir", default=TRAINING_CONFIG["output_dir"])
    parser.add_argument("--base_model", default=TRAINING_CONFIG["base_model"])
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAINING_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAINING_CONFIG["lr"])
    parser.add_argument("--val_split", type=float, default=TRAINING_CONFIG["val_split"])
    parser.add_argument("--max_len", type=int, default=TRAINING_CONFIG["max_len"])
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        max_len=args.max_len,
    )
