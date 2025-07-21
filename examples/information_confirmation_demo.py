#!/usr/bin/env python3
"""
Demo script for the new information confirmation layer in CiteWeave's multi-agent research system.

This demonstrates how the system now shows users what information has been gathered
and asks for confirmation before providing the final answer.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.multi_agent_research_system import LangGraphResearchSystem
import logging

def demo_information_confirmation():
    """Demonstrate the new information confirmation workflow"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🤖 CiteWeave Information Confirmation Demo")
    print("=" * 60)
    print("This demo shows the new layer that summarizes gathered information")
    print("and asks for user confirmation before providing the final answer.")
    print("=" * 60)
    
    # Initialize the research system
    system = LangGraphResearchSystem()
    
    # Example questions to test
    test_questions = [
        "What papers cite Rivkin's work on strategy?",
        "What does Porter's competitive strategy paper discuss?",
        "Find papers about business model innovation"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 Test {i}: {question}")
        print("-" * 50)
        
        # Step 1: Show information gathering summary and ask for confirmation
        print("🔍 Step 1: Gathering information and creating summary...")
        confirmation_request = system.research_question_with_confirmation(question)
        
        print("\n📊 Information Summary & Confirmation Request:")
        print(confirmation_request)
        
        # Step 2: Simulate user response (in real usage, this would be user input)
        print("\n👤 Step 2: Simulating user response...")
        user_response = "continue"  # Could be "continue", "expand", or "refine"
        print(f"User response: {user_response}")
        
        # Step 3: Continue with user's choice
        print("\n🚀 Step 3: Continuing with user's choice...")
        final_response = system.continue_with_confirmation(question, user_response)
        
        print("\n✅ Final Response:")
        print(final_response)
        
        print("\n" + "="*60)
        
        # Ask if user wants to continue with next test
        if i < len(test_questions):
            input("\nPress Enter to continue to next test...")

def demo_interactive_mode():
    """Demonstrate interactive mode where user can actually choose"""
    
    print("\n🎯 Interactive Demo Mode")
    print("=" * 40)
    print("In this mode, you can actually choose your response!")
    
    system = LangGraphResearchSystem()
    
    while True:
        # Get user question
        question = input("\n❓ Enter your research question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        # Step 1: Show information summary
        print("\n🔍 Gathering information...")
        confirmation_request = system.research_question_with_confirmation(question)
        
        print("\n📊 Information Summary:")
        print(confirmation_request)
        
        # Step 2: Get user choice
        print("\n👤 What would you like to do?")
        print("1. Continue - Generate final answer with current information")
        print("2. Expand - Search for additional information")
        print("3. Refine - Modify the search approach")
        
        user_choice = input("\nEnter your choice (1/2/3 or 'continue'/'expand'/'refine'): ").strip().lower()
        
        # Step 3: Continue with user's choice
        print("\n🚀 Processing your choice...")
        final_response = system.continue_with_confirmation(question, user_choice)
        
        print("\n✅ Final Response:")
        print(final_response)

if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Automated demo (shows all features)")
    print("2. Interactive demo (you choose responses)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        demo_interactive_mode()
    else:
        demo_information_confirmation() 