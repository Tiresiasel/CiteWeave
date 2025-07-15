# argument_classifier/train.py
"""
Fine-tune SciBERT (or any BERT) on citation-intent datasets
(SciCite, ACL-ARC, S2ORC) for argument / citation-relation classification.

Usage:
    python train.py --dataset scicite \
                    --model allenai/scibert_scivocab_uncased \
                    --epochs 3 \
                    --bsz 16 \
                    --lr 2e-5 \
                    --out_dir checkpoints/scicite_sciBERT
"""

import argparse, os, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score

LABEL_MAPS = {
    # dataset_name : ([label list], field_names)
    "scicite": (["background", "method", "result"], ("string", "citation_label")),
    "acl_arc": (["background", "compare", "extends", "future", "motivation",
                 "uses"], ("sentence", "label")),
    "citation_intent": (["background", "method", "result"], ("text", "intent"))
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=LABEL_MAPS.keys(),
                    help="Dataset key: scicite | acl_arc | citation_intent")
    ap.add_argument("--model", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out_dir", default="checkpoints/tmp")
    ap.add_argument("--max_len", type=int, default=256)
    return ap.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}

def main():
    args = parse_args()
    labels, (text_field, label_field) = LABEL_MAPS[args.dataset]

    # 1) load dataset
    dataset = load_dataset(args.dataset)
    id2label = {i:l for i,l in enumerate(labels)}
    label2id = {l:i for i,l in id2label.items()}

    # 2) tokenizer + label mapping
    tok = AutoTokenizer.from_pretrained(args.model)
    def tokenize(batch):
        batch["label"] = [label2id[l] for l in batch[label_field]]
        return tok(batch[text_field],
                   padding=False,
                   truncation=True,
                   max_length=args.max_len)
    ds_tok = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

    # 3) model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # 4) trainer
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"] if "validation" in ds_tok else ds_tok["test"],
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics
    )

    # 5) train
    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)

    # 6) evaluate on test
    eval_ds = ds_tok["test"] if "test" in ds_tok else ds_tok["validation"]
    print(trainer.evaluate(eval_ds))

if __name__ == "__main__":
    main()