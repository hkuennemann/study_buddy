"""
Study Buddy - Main Execution Script

This module contains the main execution logic for the Study Buddy application.
It orchestrates the entire workflow from document loading to answer generation.

The main workflow includes:
1. Environment setup and configuration validation
2. Document loading and intelligent chunking
3. AI-powered question generation
4. Vector store creation for knowledge retrieval
5. Answer generation using retrieval-augmented generation (RAG)

Key Functions:
    - setup_environment(): Configures environment and validates settings
    - main(): Orchestrates the complete Study Buddy workflow

Dependencies:
    - src.load_and_split: Document loading and chunking
    - src.generating_questions: AI question generation
    - src.generating_answers: Answer generation with RAG
    - dotenv: Environment variable management
    - pathlib: File system operations

Example:
    python main.py

Environment Variables:
    - FILE_PATH: Path to the PDF file to process (required)
    - QUESTION_LIMIT: Maximum number of questions to answer (optional)
    - OPENAI_API_KEY: OpenAI API key (optional)
    - GEMINI_API_KEY: Google Gemini API key (optional)
"""

# ------------------------------------------------------------
#   Study Buddy - Main Execution Script
# ------------------------------------------------------------

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from src import *

# ------------------------------------------------------------
#   Setup and Configuration
# ------------------------------------------------------------

def setup_environment():
    """
    Set up environment and create necessary directories.
    
    This function loads environment variables from .env file, creates the outputs
    directory if it doesn't exist, and validates required environment variables.
    It also handles optional configuration for question limits.
    
    Returns:
        tuple: A tuple containing (file_path, question_limit) where:
            - file_path (str): Path to the PDF file to process
            - question_limit (int or None): Maximum number of questions to answer,
              or None if no limit is set
    
    Raises:
        SystemExit: If required environment variables are missing or invalid
    """
    # Load .env file
    load_dotenv()
    
    # Create outputs directory if it doesn't exist
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Verify required environment variables
    required_vars = ["FILE_PATH"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these in your .env file")
        sys.exit(1)
    
    # Get optional limit for questions
    question_limit = os.getenv("QUESTION_LIMIT")
    if question_limit:
        try:
            question_limit = int(question_limit)
        except ValueError:
            print("⚠️  Invalid QUESTION_LIMIT value, ignoring...")
            question_limit = None
    else:
        question_limit = None
    
    return os.getenv("FILE_PATH"), question_limit

def main():
    """
    Main execution function for Study Buddy.
    
    This function orchestrates the entire Study Buddy workflow:
    1. Sets up the environment and validates configuration
    2. Loads and splits the PDF document into chunks
    3. Generates study questions using AI
    4. Creates a vector store for answer generation
    5. Generates answers to the questions and saves them to file
    
    The function handles errors gracefully and provides progress updates
    throughout the process.
    
    Returns:
        None: This function doesn't return a value but prints progress
        and saves results to outputs/answers.txt
    
    Raises:
        SystemExit: If critical errors occur during processing
    """

    print("🧠 Study Buddy - Starting...")
    
    # Setup environment
    file_path, question_limit = setup_environment()
    
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("Please check your FILE_PATH in .env file")
        sys.exit(1)
    
    print(f"(1) 📄 Processing file: {os.path.basename(file_path)}")
    
    # ------------------------------------------------------------
    #   Load and split documents
    # ------------------------------------------------------------
    
    try:
        docs_q, docs_a = load_and_split(file_path, suppress_warnings=True)
        print(f"(2) 📚 Loaded {len(docs_q)} question chunks and {len(docs_a)} answer chunks")
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        sys.exit(1)
    
    # ------------------------------------------------------------
    #   Generate questions
    # ------------------------------------------------------------
    
    try:
        questions_text = generate_questions(docs_q, provider="gemini")
        # Count questions
        question_lines = [line.strip() for line in questions_text.split('\n') if re.match(r'^\d+\.', line.strip())]
        question_count = len(question_lines)
        print(f"(3) 📝 Generated {question_count} questions")
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        sys.exit(1)
    
    # ------------------------------------------------------------
    #   Generate answers
    # ------------------------------------------------------------
    
    try:
        vector_store = create_vector_store(docs_a, "gemini")
        print("(4) 🔍 Vector store created successfully")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        sys.exit(1)
    
    try:
        retrieve_answers(
            questions=questions_text,
            vector_store=vector_store, 
            provider="gemini",
            limit=question_limit
        )
        print("(5) 📁 Answers generated successfully and results saved to outputs/answers.txt")
    except Exception as e:
        print(f"❌ Error generating answers: {e}")
        sys.exit(1)
    
    print("✅ Done!")

if __name__ == "__main__":
    main()
