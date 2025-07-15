#!/usr/bin/env python3
# argument_classifier/run_example.py
"""
Example script demonstrating the complete workflow for citation-argument relation classification.

This script shows how to:
1. Train a model with automatic data splitting
2. Test the trained model on test data
3. Generate performance reports

Usage:
    python run_example.py
"""

import os
import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    result = subprocess.run(command, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Error: {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"✅ {description} completed successfully!")
        return True


def main():
    """Run the complete example workflow."""
    print("🎯 Citation-Argument Relation Classification - Complete Workflow Example")
    print("=" * 80)
    
    # Check if data files exist
    if not os.path.exists("datasets/single_relation.jsonl") or not os.path.exists("datasets/multi_relation.jsonl"):
        print("❌ Error: Data files not found!")
        print("Please ensure single_relation.jsonl and multi_relation.jsonl are in the datasets/ directory.")
        return
    
    print("📂 Data files found:")
    print("  ✓ datasets/single_relation.jsonl")
    print("  ✓ datasets/multi_relation.jsonl")
    
    # Step 1: Train the model
    train_command = """python training.py \
        --single_data datasets/single_relation.jsonl \
        --multi_data datasets/multi_relation.jsonl \
        --epochs 3 \
        --batch_size 8 \
        --learning_rate 2e-5 \
        --output_dir checkpoints/example_model \
        --train_ratio 0.8"""
    
    if not run_command(train_command, "Training Citation-Argument Classifier"):
        return
    
    # Step 2: Test the model
    test_command = """python test.py \
        --model_dir checkpoints/example_model \
        --test_data datasets/test_data.jsonl \
        --output_report example_test_results.json \
        --batch_size 16"""
    
    # Note: We need to find the correct checkpoint directory
    checkpoint_dirs = []
    model_dir = "checkpoints/example_model"
    if os.path.exists(model_dir):
        for item in os.listdir(model_dir):
            if item.startswith("checkpoint-"):
                checkpoint_dirs.append(item)
    
    if checkpoint_dirs:
        # Use the latest checkpoint
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
        checkpoint_path = f"{model_dir}/{latest_checkpoint}"
        
        test_command = f"""python test.py \
            --model_dir {checkpoint_path} \
            --test_data datasets/test_data.jsonl \
            --output_report example_test_results.json \
            --batch_size 16"""
        
        if not run_command(test_command, "Testing Trained Model"):
            return
    else:
        print("❌ Error: No checkpoint found in model directory")
        return
    
    # Step 3: Display results summary
    print(f"\n{'='*60}")
    print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    print("\n📁 Generated Files:")
    files_to_check = [
        ("datasets/train_data.jsonl", "Training data (80% of original)"),
        ("datasets/test_data.jsonl", "Test data (20% of original)"),
        (f"{checkpoint_path}/", "Trained model checkpoint"),
        ("example_test_results.json", "Detailed test results"),
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path} - {description}")
        else:
            print(f"  ❌ {file_path} - {description} (not found)")
    
    print(f"\n📊 Quick Performance Summary:")
    try:
        import json
        with open("example_test_results.json", 'r') as f:
            results = json.load(f)
        
        token_acc = results['token_level_metrics']['accuracy']
        entity_f1 = results['entity_level_metrics']['f1']
        test_count = results['test_examples_count']
        
        print(f"  • Test Examples: {test_count}")
        print(f"  • Token Accuracy: {token_acc:.1%}")
        print(f"  • Entity F1-Score: {entity_f1:.1%}")
        print(f"  • Device Used: {results['device_used']}")
        
    except Exception as e:
        print(f"  ❌ Could not read test results: {e}")
    
    print(f"\n🎯 Next Steps:")
    print("  1. Review detailed results in 'example_test_results.json'")
    print("  2. Use 'inference.py' for real-time prediction")
    print("  3. Integrate trained model with your CiteWeave pipeline")
    print(f"  4. Model is saved at: {checkpoint_path}")
    
    print(f"\n💡 Usage Tips:")
    print("  • Adjust --epochs, --batch_size, and --learning_rate for different performance")
    print("  • Use --train_ratio to change train/test split ratio")
    print("  • The model automatically optimizes for your hardware (MPS/CUDA/CPU)")


if __name__ == "__main__":
    main() 