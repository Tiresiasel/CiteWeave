# Argument / Citation-Intent Classifier

This module fine-tunes **SciBERT** (or any BERT-family model) on public
citation-intent datasets (SciCite, ACL-ARC, SemanticScholar) to classify
the rhetorical relationship between a citing sentence and the cited work
(e.g., *background*, *method-use*, *result-comparison*).

## Quick Start
```bash
# install deps
pip install torch transformers datasets scikit-learn tqdm

# train on SciCite with SciBERT
python train.py \
   --dataset scicite \
   --model allenai/scibert_scivocab_uncased \
   --epochs 3 \
   --bsz 16 \
   --lr 2e-5 \
   --out_dir checkpoints/scicite_sciBERT