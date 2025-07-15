# argument_classifier/data_utils.py

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class CitationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx])
        }

def load_data(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    return df["text"].tolist(), df["label"].tolist()