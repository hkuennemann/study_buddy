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
    """Set up environment and create necessary directories."""
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
    
    return os.getenv("FILE_PATH")

def main():
    """Main execution function for Study Buddy."""

    print("🧠 Study Buddy - Starting...")
    
    # Setup environment
    file_path = setup_environment()
    
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
            provider="gemini"
        )
        print("(5) 📁 Answers generated successfully and results saved to outputs/answers.txt")
    except Exception as e:
        print(f"❌ Error generating answers: {e}")
        sys.exit(1)
    
    print("✅ Done!")

if __name__ == "__main__":
    main()
