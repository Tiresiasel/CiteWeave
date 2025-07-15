MODEL_NAME = "allenai/scibert_scivocab_uncased"

RELATION_LABELS = [
    "CITES",
    "SUPPORTS",
    "REFUTES",
    "ELABORATES",
    "QUALIFIES",
    "EXTENDS",
    "CONTRASTS",
    "REPEATS",
    "INSPIRED_BY",
    "USES_METHOD_OF",
]

LABEL2ID = {label: i for i, label in enumerate(RELATION_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}