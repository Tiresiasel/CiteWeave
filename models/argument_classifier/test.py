# argument_classifier/test.py
"""
Test script for the citation-argument relation classification model.
Evaluates the trained model on test data and provides detailed performance metrics.

Usage:
    python test.py --model_dir checkpoints/citation_classifier \
                   --test_data datasets/test_data.jsonl \
                   --output_report test_results.json
"""

import argparse
import json
import os
import numpy as np
import torch
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)

from training import (
    CitationExample, 
    load_citation_data, 
    align_labels_with_tokens,
    RELATION_LABELS, 
    LABEL_TO_ID, 
    ID_TO_LABEL
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="checkpoints/citation_classifier",
                       help="Path to trained model directory")
    parser.add_argument("--test_data", default="datasets/test_data.jsonl",
                       help="Path to test data JSONL file")
    parser.add_argument("--output_report", default="test_results.json",
                       help="Path to save test results")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None,
                       help="Device to use (auto-detect if None)")
    return parser.parse_args()


def setup_device(device_arg=None):
    """Setup device for inference."""
    if device_arg:
        device = torch.device(device_arg)
        print(f"🔧 Using specified device: {device}")
        return device
    
    # Auto-detect best device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("🍎 Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    return device


def prepare_test_batch(examples: List[CitationExample], tokenizer, max_length: int):
    """Prepare a batch of examples for inference."""
    texts = [example.text for example in examples]
    
    # Tokenize texts
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Create ground truth labels
    all_labels = []
    for example in examples:
        # Create aligned labels for this example
        encoding = tokenizer(
            example.text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        
        aligned_labels = align_labels_with_tokens(
            example.text, example.relations, encoding, tokenizer, max_length
        )
        all_labels.append(aligned_labels)
    
    return encodings, torch.tensor(all_labels)


def extract_entity_predictions(predictions, labels, attention_mask):
    """Extract entity-level predictions from token-level predictions."""
    entity_predictions = []
    entity_labels = []
    
    for pred_seq, label_seq, mask_seq in zip(predictions, labels, attention_mask):
        # Skip special tokens and padding
        valid_length = mask_seq.sum().item()
        pred_seq = pred_seq[1:valid_length-1]  # Skip [CLS] and [SEP]
        label_seq = label_seq[1:valid_length-1]
        
        # Extract entities from BIO tags
        pred_entities = extract_entities_from_bio(pred_seq)
        true_entities = extract_entities_from_bio(label_seq)
        
        entity_predictions.extend(pred_entities)
        entity_labels.extend(true_entities)
    
    return entity_predictions, entity_labels


def extract_entities_from_bio(bio_sequence):
    """Extract entities from BIO tag sequence."""
    entities = []
    current_entity = None
    
    for i, label_id in enumerate(bio_sequence):
        if label_id == -100:  # Skip ignored tokens
            continue
            
        label = ID_TO_LABEL[label_id]
        
        if label.startswith('B-'):
            # Start of new entity
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'start': i,
                'end': i,
                'label': label[2:]  # Remove 'B-' prefix
            }
        elif label.startswith('I-') and current_entity:
            # Continuation of entity
            if label[2:] == current_entity['label']:
                current_entity['end'] = i
            else:
                # Label mismatch, start new entity
                entities.append(current_entity)
                current_entity = {
                    'start': i,
                    'end': i,
                    'label': label[2:]
                }
        elif label == 'O':
            # Outside any entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Add the last entity if exists
    if current_entity:
        entities.append(current_entity)
    
    return entities


def calculate_entity_metrics(pred_entities, true_entities):
    """Calculate entity-level precision, recall, and F1."""
    pred_set = set((e['start'], e['end'], e['label']) for e in pred_entities)
    true_set = set((e['start'], e['end'], e['label']) for e in true_entities)
    
    if len(true_set) == 0:
        precision = 1.0 if len(pred_set) == 0 else 0.0
        recall = 1.0
        f1 = 1.0 if len(pred_set) == 0 else 0.0
    else:
        correct = len(pred_set & true_set)
        precision = correct / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = correct / len(true_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def create_confusion_matrix_text(y_true, y_pred, labels):
    """Create text representation of confusion matrix."""
    # Filter out -100 labels
    filtered_true = []
    filtered_pred = []
    
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label != -100:
            filtered_true.append(true_label)
            filtered_pred.append(pred_label)
    
    # Create confusion matrix
    cm = confusion_matrix(filtered_true, filtered_pred)
    
    # Create text representation
    matrix_text = "\nConfusion Matrix (rows=true, cols=predicted):\n"
    matrix_text += "Label".ljust(15)
    for i in range(len(RELATION_LABELS)):
        matrix_text += f"{i:>4}"
    matrix_text += "\n"
    
    for i, label in enumerate(RELATION_LABELS):
        matrix_text += f"{label:<15}"
        for j in range(len(RELATION_LABELS)):
            if i < len(cm) and j < len(cm[i]):
                matrix_text += f"{cm[i][j]:>4}"
            else:
                matrix_text += f"{0:>4}"
        matrix_text += "\n"
    
    return matrix_text


def evaluate_model(model, tokenizer, test_examples, device, max_length, batch_size):
    """Evaluate model on test data."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_entity_predictions = []
    all_entity_labels = []
    
    print(f"🧪 Evaluating on {len(test_examples)} test examples...")
    
    # Process in batches
    for i in range(0, len(test_examples), batch_size):
        batch_examples = test_examples[i:i+batch_size]
        
        # Prepare batch
        encodings, labels = prepare_test_batch(batch_examples, tokenizer, max_length)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        labels = labels.to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Move to CPU for processing
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        attention_mask = encodings['attention_mask'].cpu().numpy()
        
        # Collect token-level predictions
        for pred_seq, label_seq, mask_seq in zip(predictions, labels, attention_mask):
            valid_length = mask_seq.sum()
            for j in range(valid_length):
                if label_seq[j] != -100:  # Skip ignored tokens
                    all_predictions.append(pred_seq[j])
                    all_labels.append(label_seq[j])
        
        # Collect entity-level predictions
        entity_preds, entity_labels = extract_entity_predictions(predictions, labels, attention_mask)
        all_entity_predictions.extend(entity_preds)
        all_entity_labels.extend(entity_labels)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch_examples)}/{len(test_examples)} examples...")
    
    return all_predictions, all_labels, all_entity_predictions, all_entity_labels


def main():
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Load test data
    print(f"📂 Loading test data from {args.test_data}")
    if not os.path.exists(args.test_data):
        print(f"❌ Test data file not found: {args.test_data}")
        print("💡 Run training.py first to generate test data")
        return
    
    test_examples = load_citation_data(args.test_data)
    print(f"Loaded {len(test_examples)} test examples")
    
    # Load model and tokenizer
    print(f"🤖 Loading model from {args.model_dir}")
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory not found: {args.model_dir}")
        print("💡 Run training.py first to train the model")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(device)
    
    # Load model configuration
    config_path = os.path.join(args.model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_config = json.load(f)
            print(f"📋 Model trained on {model_config.get('device_used', 'unknown')} device")
    
    # Evaluate model
    print("\n🔬 Starting evaluation...")
    predictions, labels, entity_predictions, entity_labels = evaluate_model(
        model, tokenizer, test_examples, device, args.max_length, args.batch_size
    )
    
    # Calculate token-level metrics
    print("\n📊 Calculating token-level metrics...")
    token_accuracy = accuracy_score(labels, predictions)
    token_f1_macro = f1_score(labels, predictions, average='macro')
    token_f1_weighted = f1_score(labels, predictions, average='weighted')
    token_precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    token_recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    
    # Calculate entity-level metrics
    print("📊 Calculating entity-level metrics...")
    entity_precision, entity_recall, entity_f1 = calculate_entity_metrics(entity_predictions, entity_labels)
    
    # Per-class metrics
    target_names = [ID_TO_LABEL[i] for i in range(len(RELATION_LABELS))]
    class_report = classification_report(labels, predictions, target_names=target_names, 
                                       zero_division=0, output_dict=True)
    
    # Relation type analysis
    relation_stats = defaultdict(lambda: {'predicted': 0, 'actual': 0, 'correct': 0})
    
    for pred_entity in entity_predictions:
        relation_stats[pred_entity['label']]['predicted'] += 1
    
    for true_entity in entity_labels:
        relation_stats[true_entity['label']]['actual'] += 1
    
    pred_set = set((e['start'], e['end'], e['label']) for e in entity_predictions)
    true_set = set((e['start'], e['end'], e['label']) for e in entity_labels)
    correct_entities = pred_set & true_set
    
    for start, end, label in correct_entities:
        relation_stats[label]['correct'] += 1
    
    # Calculate per-relation metrics
    per_relation_metrics = {}
    for relation in relation_stats:
        stats = relation_stats[relation]
        prec = stats['correct'] / stats['predicted'] if stats['predicted'] > 0 else 0.0
        rec = stats['correct'] / stats['actual'] if stats['actual'] > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        per_relation_metrics[relation] = {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'predicted_count': stats['predicted'],
            'actual_count': stats['actual'],
            'correct_count': stats['correct']
        }
    
    # Print results
    print("\n" + "="*60)
    print("🎯 TEST RESULTS")
    print("="*60)
    
    print(f"\n📊 Token-Level Metrics:")
    print(f"  Accuracy: {token_accuracy:.4f} ({token_accuracy:.1%})")
    print(f"  F1-Score (Macro): {token_f1_macro:.4f} ({token_f1_macro:.1%})")
    print(f"  F1-Score (Weighted): {token_f1_weighted:.4f} ({token_f1_weighted:.1%})")
    print(f"  Precision (Macro): {token_precision_macro:.4f} ({token_precision_macro:.1%})")
    print(f"  Recall (Macro): {token_recall_macro:.4f} ({token_recall_macro:.1%})")
    
    print(f"\n🎯 Entity-Level Metrics:")
    print(f"  Precision: {entity_precision:.4f} ({entity_precision:.1%})")
    print(f"  Recall: {entity_recall:.4f} ({entity_recall:.1%})")
    print(f"  F1-Score: {entity_f1:.4f} ({entity_f1:.1%})")
    print(f"  Predicted Entities: {len(entity_predictions)}")
    print(f"  Actual Entities: {len(entity_labels)}")
    print(f"  Correct Entities: {len(correct_entities)}")
    
    print(f"\n🏷️  Per-Relation Performance:")
    for relation, metrics in per_relation_metrics.items():
        print(f"  {relation}:")
        print(f"    Precision: {metrics['precision']:.4f} ({metrics['precision']:.1%})")
        print(f"    Recall: {metrics['recall']:.4f} ({metrics['recall']:.1%})")
        print(f"    F1-Score: {metrics['f1']:.4f} ({metrics['f1']:.1%})")
        print(f"    Predicted/Actual/Correct: {metrics['predicted_count']}/{metrics['actual_count']}/{metrics['correct_count']}")
    
    # Create confusion matrix text
    confusion_matrix_text = create_confusion_matrix_text(labels, predictions, target_names)
    print(confusion_matrix_text)
    
    # Prepare results for saving
    results = {
        'test_data_path': args.test_data,
        'model_path': args.model_dir,
        'test_examples_count': len(test_examples),
        'device_used': str(device),
        
        'token_level_metrics': {
            'accuracy': float(token_accuracy),
            'f1_macro': float(token_f1_macro),
            'f1_weighted': float(token_f1_weighted),
            'precision_macro': float(token_precision_macro),
            'recall_macro': float(token_recall_macro)
        },
        
        'entity_level_metrics': {
            'precision': float(entity_precision),
            'recall': float(entity_recall),
            'f1': float(entity_f1),
            'predicted_entities': len(entity_predictions),
            'actual_entities': len(entity_labels),
            'correct_entities': len(correct_entities)
        },
        
        'per_relation_metrics': per_relation_metrics,
        'classification_report': class_report,
        'confusion_matrix_text': confusion_matrix_text
    }
    
    # Save results
    print(f"\n💾 Saving results to {args.output_report}")
    with open(args.output_report, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Testing completed!")
    print(f"📋 Detailed results saved to {args.output_report}")
    
    # Summary
    high_performing = [r for r, m in per_relation_metrics.items() if m['f1'] > 0.8]
    print(f"\n🎉 Summary:")
    print(f"  Overall Token Accuracy: {token_accuracy:.1%}")
    print(f"  Entity F1-Score: {entity_f1:.1%}")
    print(f"  High Performing Relations (F1>80%): {', '.join(high_performing) if high_performing else 'None'}")


if __name__ == "__main__":
    main() 