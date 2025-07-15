# argument_classifier/inference.py
"""
Inference script for citation-argument relation classification.
Uses the trained model to identify citation spans and their relations in new text.

Usage:
    from inference import CitationRelationClassifier
    
    classifier = CitationRelationClassifier("checkpoints/citation_classifier")
    relations = classifier.predict("Porter (2006) extends earlier frameworks by introducing multi-level dynamics.")
    print(relations)  # [{"span": "Porter (2006)", "label": "EXTENDS", "confidence": 0.95}]
"""

import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re


class CitationRelationClassifier:
    """Citation-argument relation classifier for inference."""
    
    def __init__(self, model_path: str):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = model_path
        
        # Set device with Apple Silicon MPS support
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🍎 Using Apple Silicon MPS acceleration for inference")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🚀 Using CUDA acceleration for inference")
        else:
            self.device = torch.device("cpu")
            print("💻 Using CPU for inference")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load configuration - try training config first, then use model config
        try:
            with open(f"{model_path}/training_config.json", 'r') as f:
                self.training_config = json.load(f)
            self.id_to_label = self.training_config['id_to_label']
            self.label_to_id = self.training_config['label_to_id']
            self.relation_labels = self.training_config['relation_labels']
        except FileNotFoundError:
            # Fallback to model's built-in label mappings
            self.id_to_label = self.model.config.id2label
            self.label_to_id = self.model.config.label2id
            self.relation_labels = list(self.model.config.id2label.values())
            self.training_config = None
        
    def predict(self, text: str, return_confidence: bool = True) -> List[Dict[str, any]]:
        """
        Predict citation relations in the given text.
        
        Args:
            text: Input text containing citations
            return_confidence: Whether to include confidence scores
            
        Returns:
            List of dictionaries with 'span', 'label', and optionally 'confidence'
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.training_config['max_length'] if self.training_config else 512,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop('offset_mapping')[0].numpy()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
            confidences = torch.max(predictions, dim=-1)[0][0].cpu().numpy()
        
        # Convert predictions to spans
        relations = self._extract_spans_from_predictions(
            text, predicted_labels, confidences, offset_mapping, return_confidence
        )
        
        return relations
    
    def _extract_spans_from_predictions(
        self, 
        text: str, 
        predicted_labels: np.ndarray, 
        confidences: np.ndarray,
        offset_mapping: np.ndarray,
        return_confidence: bool
    ) -> List[Dict[str, any]]:
        """Extract citation spans from BIO predictions."""
        relations = []
        current_span = None
        current_label = None
        current_confidence_scores = []
        
        for idx, (label_id, confidence) in enumerate(zip(predicted_labels, confidences)):
            if idx >= len(offset_mapping):
                break
                
            start, end = offset_mapping[idx]
            if start is None or end is None:  # Special tokens
                continue
                
            # Handle both int and numpy int types
            try:
                label = self.id_to_label[int(label_id)]
            except KeyError:
                label = self.id_to_label[label_id]
            
            if label.startswith('B-'):
                # Start of new entity
                if current_span is not None:
                    # Finish previous span
                    relations.append(self._create_relation_dict(
                        text, current_span, current_label, current_confidence_scores, return_confidence
                    ))
                
                # Start new span
                relation_type = label[2:]  # Remove 'B-' prefix
                current_span = (start, end)
                current_label = relation_type
                current_confidence_scores = [confidence]
                
            elif label.startswith('I-') and current_span is not None:
                # Continue current entity
                relation_type = label[2:]  # Remove 'I-' prefix
                if relation_type == current_label:
                    # Extend current span
                    current_span = (current_span[0], end)
                    current_confidence_scores.append(confidence)
                else:
                    # Different label, finish previous and start new
                    relations.append(self._create_relation_dict(
                        text, current_span, current_label, current_confidence_scores, return_confidence
                    ))
                    current_span = (start, end)
                    current_label = relation_type
                    current_confidence_scores = [confidence]
                    
            else:
                # 'O' label or start of sentence after entity
                if current_span is not None:
                    relations.append(self._create_relation_dict(
                        text, current_span, current_label, current_confidence_scores, return_confidence
                    ))
                    current_span = None
                    current_label = None
                    current_confidence_scores = []
        
        # Handle last span
        if current_span is not None:
            relations.append(self._create_relation_dict(
                text, current_span, current_label, current_confidence_scores, return_confidence
            ))
        
        return relations
    
    def _create_relation_dict(
        self, 
        text: str, 
        span: Tuple[int, int], 
        label: str, 
        confidence_scores: List[float],
        return_confidence: bool
    ) -> Dict[str, any]:
        """Create a relation dictionary from span information."""
        start, end = span
        span_text = text[start:end].strip()
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        result = {
            "span": span_text,
            "label": label,
            "start": start,
            "end": end
        }
        
        if return_confidence:
            result["confidence"] = float(avg_confidence)
            
        return result
    
    def predict_batch(self, texts: List[str], return_confidence: bool = True) -> List[List[Dict[str, any]]]:
        """
        Predict citation relations for a batch of texts.
        
        Args:
            texts: List of input texts
            return_confidence: Whether to include confidence scores
            
        Returns:
            List of prediction lists, one for each input text
        """
        return [self.predict(text, return_confidence) for text in texts]
    
    def analyze_text(self, text: str) -> Dict[str, any]:
        """
        Comprehensive analysis of citation relations in text.
        
        Args:
            text: Input text containing citations
            
        Returns:
            Dictionary with detailed analysis including:
            - relations: List of detected relations
            - summary: Summary statistics
            - citation_patterns: Analysis of citation patterns
        """
        relations = self.predict(text, return_confidence=True)
        
        # Generate summary statistics
        relation_counts = {}
        total_confidence = 0
        
        for relation in relations:
            label = relation['label']
            relation_counts[label] = relation_counts.get(label, 0) + 1
            total_confidence += relation['confidence']
        
        avg_confidence = total_confidence / len(relations) if relations else 0
        
        # Analyze citation patterns
        citation_patterns = self._analyze_citation_patterns(text, relations)
        
        return {
            'text': text,
            'relations': relations,
            'summary': {
                'total_citations': len(relations),
                'relation_types': len(relation_counts),
                'relation_distribution': relation_counts,
                'average_confidence': avg_confidence
            },
            'citation_patterns': citation_patterns
        }
    
    def _analyze_citation_patterns(self, text: str, relations: List[Dict]) -> Dict[str, any]:
        """Analyze patterns in the detected citations."""
        patterns = {
            'parenthetical_citations': 0,
            'narrative_citations': 0,
            'multi_author_citations': 0,
            'year_range_citations': 0
        }
        
        for relation in relations:
            span = relation['span']
            
            # Check for parenthetical vs narrative
            if span.startswith('(') and span.endswith(')'):
                patterns['parenthetical_citations'] += 1
            else:
                patterns['narrative_citations'] += 1
            
            # Check for multiple authors
            if ' and ' in span or ', ' in span or ' et al.' in span:
                patterns['multi_author_citations'] += 1
            
            # Check for year ranges
            if re.search(r'\d{4}-\d{4}', span):
                patterns['year_range_citations'] += 1
        
        return patterns


def demo_usage():
    """Demonstrate how to use the classifier."""
    
    # Example texts for testing
    test_texts = [
        "Porter (2006) extends earlier frameworks by introducing multi-level dynamics into dynamic capabilities.",
        "The theoretical framing is inspired by concepts loosely aligned with Mintzberg, 2007.",
        "However, the analysis presented by Ghemawat (2019) refutes this claim, especially under high uncertainty.",
        "Grant 2001 extends earlier frameworks. Prior research by Grant, 2019 laid the foundation. Empirical findings in Porter 1980 strongly support the argument.",
    ]
    
    print("Citation Relation Classifier Demo")
    print("=" * 50)
    
    # Note: This assumes you have a trained model
    try:
        classifier = CitationRelationClassifier("checkpoints/citation_classifier")
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nExample {i}:")
            print(f"Text: {text}")
            
            # Basic prediction
            relations = classifier.predict(text)
            print(f"Relations: {relations}")
            
            # Detailed analysis
            analysis = classifier.analyze_text(text)
            print(f"Summary: {analysis['summary']}")
            
    except FileNotFoundError:
        print("Model not found. Please train the model first using training.py")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    demo_usage() 