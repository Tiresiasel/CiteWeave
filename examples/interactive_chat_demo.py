#!/usr/bin/env python3
"""
Demo script for the new interactive chat functionality in CiteWeave's multi-agent research system.

This demonstrates the true interactive chat process where the system:
1. Gathers information
2. Shows you what it found
3. Asks if it's enough
4. If not, asks what additional information you want
5. Gathers more information based on your instructions
6. Repeats until you're satisfied
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.multi_agent_research_system import LangGraphResearchSystem
import logging

def main():
    """Main demo function"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🤖 CiteWeave Interactive Chat Demo")
    print("=" * 60)
    print("Interactive Research Chat - The system will:")
    print("• Gather information based on your question")
    print("• Show you what it found")
    print("• Ask if you need more information")
    print("• Gather additional data based on your instructions")
    print("• Continue until you're satisfied")
    print("=" * 60)
    
    # Initialize the research system
    system = LangGraphResearchSystem()
    
    # Get user question
    question = input("\n❓ Enter your research question: ").strip()
    
    if not question:
        question = "What papers cite Rivkin's work on strategy?"
        print(f"Using default question: {question}")
    
    print(f"\n🚀 Starting interactive chat research...")
    print("The system will now gather information and ask for your input!")
    print("-" * 60)
    
    # Start the interactive chat
    result = system.interactive_research_chat(question)
    
    print("\n" + "=" * 60)
    print("🎉 Interactive research completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 