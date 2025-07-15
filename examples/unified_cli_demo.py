#!/usr/bin/env python3
"""
Unified CLI Demo
Shows how to use the new unified CiteWeave CLI interface.
"""

import subprocess
import sys
import os

def run_cli_command(cmd_args):
    """Run a CLI command and return the result."""
    try:
        result = subprocess.run([sys.executable, "-m", "src.cli"] + cmd_args, 
                              capture_output=True, text=True, cwd=os.getcwd())
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def demo_unified_cli():
    """Demonstrate the unified CLI capabilities."""
    print("🚀 CiteWeave Unified CLI Demo")
    print("=" * 50)
    
    # Demo 1: Show help
    print("\n1. CLI Help:")
    print("-" * 20)
    code, stdout, stderr = run_cli_command(["--help"])
    if code == 0:
        print(stdout)
    else:
        print(f"Error: {stderr}")
    
    # Demo 2: Show database status  
    print("\n2. Database Status:")
    print("-" * 20)
    code, stdout, stderr = run_cli_command(["status"])
    if code == 0:
        print(stdout)
    else:
        print(f"Status command failed (this is expected if database is not set up)")
        print(f"Error: {stderr}")
    
    # Demo 3: List documents
    print("\n3. List Available Documents:")
    print("-" * 30)
    code, stdout, stderr = run_cli_command(["list"])
    if code == 0:
        print(stdout)
    else:
        print(f"List command failed (this is expected if no documents are available)")
        print(f"Error: {stderr}")
    
    # Demo 4: Show query help
    print("\n4. Query Options:")
    print("-" * 20)
    code, stdout, stderr = run_cli_command(["query", "--help"])
    if code == 0:
        print(stdout)
    else:
        print(f"Error: {stderr}")
    
    # Demo 5: Show network help
    print("\n5. Network Analysis Options:")
    print("-" * 30)
    code, stdout, stderr = run_cli_command(["network", "--help"])
    if code == 0:
        print(stdout)
    else:
        print(f"Error: {stderr}")
    
    print("\n✅ Demo completed!")
    print("\nKey CLI Features:")
    print("• Document processing: upload, diagnose")
    print("• Database management: status, import, list")
    print("• Query capabilities: cypher queries, semantic search")
    print("• Citation network analysis: stats, stub papers")
    print("• Database operations: reset")

if __name__ == "__main__":
    demo_unified_cli() 