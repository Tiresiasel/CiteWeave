#!/usr/bin/env python3
"""
Batch Upload Demo
Demonstrates the batch upload functionality of CiteWeave CLI.
"""

import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path

def create_demo_pdfs():
    """Create a demo directory structure with some sample PDFs."""
    demo_dir = Path("demo_batch_pdfs")
    
    # Remove existing demo directory if it exists
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    
    # Create directory structure
    demo_dir.mkdir()
    (demo_dir / "subfolder").mkdir()
    
    # Copy sample PDFs if they exist in test_files
    test_files_dir = Path("test_files")
    if test_files_dir.exists():
        pdf_files = list(test_files_dir.glob("*.pdf"))
        if pdf_files:
            # Copy first PDF to main directory
            if len(pdf_files) >= 1:
                shutil.copy2(pdf_files[0], demo_dir)
            
            # Copy second PDF to subfolder if available
            if len(pdf_files) >= 2:
                shutil.copy2(pdf_files[1], demo_dir / "subfolder")
            
            print(f"✅ Created demo directory with {len(pdf_files)} PDF files")
            return demo_dir
    
    print("❌ No sample PDF files found in test_files directory")
    return None

def run_cli_command(cmd_args):
    """Run a CLI command and return the result."""
    try:
        result = subprocess.run([sys.executable, "-m", "src.cli"] + cmd_args, 
                              capture_output=True, text=True, cwd=os.getcwd())
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def demo_batch_upload():
    """Demonstrate batch upload functionality."""
    print("🚀 CiteWeave Batch Upload Demo")
    print("=" * 50)
    
    # Create demo directory
    demo_dir = create_demo_pdfs()
    if not demo_dir:
        return
    
    try:
        print(f"\n📁 Demo directory structure:")
        for item in demo_dir.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(demo_dir)
                print(f"   📄 {relative_path}")
        
        print(f"\n1. Basic batch upload (sequential processing):")
        print("-" * 40)
        code, stdout, stderr = run_cli_command([
            "batch-upload", str(demo_dir), "--continue-on-error"
        ])
        
        if code == 0:
            print("✅ Batch upload completed successfully")
            print("Output summary:")
            # Extract key lines from output
            lines = stdout.split('\n')
            for line in lines:
                if ('📚 Found' in line or 
                    '📊 Batch Processing Summary' in line or
                    '✅ Successful:' in line or
                    '❌ Failed:' in line or
                    'Total files processed:' in line):
                    print(f"   {line}")
        else:
            print(f"❌ Batch upload failed")
            print(f"Error: {stderr}")
        
        print(f"\n2. Recursive batch upload:")
        print("-" * 30)
        code, stdout, stderr = run_cli_command([
            "batch-upload", str(demo_dir), "--recursive", "--force"
        ])
        
        if code == 0:
            lines = stdout.split('\n')
            for line in lines:
                if '📚 Found' in line:
                    print(f"   {line}")
        
        print(f"\n3. Parallel processing demo:")
        print("-" * 30)
        print("   Command: python3 -m src.cli batch-upload demo_batch_pdfs --max-workers 2 --recursive")
        print("   (This would process files in parallel with 2 workers)")
        
        print(f"\n4. Quality diagnosis demo:")
        print("-" * 25)
        print("   Command: python3 -m src.cli batch-upload demo_batch_pdfs --diagnose --recursive")
        print("   (This would run quality diagnosis before processing each file)")
        
    finally:
        # Clean up demo directory
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
            print(f"\n🧹 Cleaned up demo directory: {demo_dir}")
    
    print(f"\n✅ Demo completed!")
    print("\nBatch Upload Features:")
    print("• 📁 Process all PDFs in a directory")
    print("• 🔄 Recursive subdirectory scanning")
    print("• ⚡ Configurable parallel processing")
    print("• 🛡️ Error handling and continuation")
    print("• 📊 Detailed progress and statistics")
    print("• 🔍 Optional quality diagnosis")
    print("• 💪 Force reprocessing option")

if __name__ == "__main__":
    demo_batch_upload() 