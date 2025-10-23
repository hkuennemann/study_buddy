# ------------------------------------------------------------
#   Study Buddy - Main Execution Script
# ------------------------------------------------------------

import os
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
    
    print(f"📄 Processing file: {file_path}")
    
    # ------------------------------------------------------------
    #   Load and split documents
    # ------------------------------------------------------------
    
    print("📚 Loading and splitting documents...")
    try:
        docs_q, docs_a = load_and_split(file_path, suppress_warnings=True)
        print(f"✅ Loaded {len(docs_q)} question chunks and {len(docs_a)} answer chunks")
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        sys.exit(1)
    
    # ------------------------------------------------------------
    #   Generate questions
    # ------------------------------------------------------------
    
    print("❓ Generating questions...")
    try:
        questions_text = generate_questions(docs_q, provider="gemini")
        print("✅ Questions generated successfully")
        question_count = len(questions_text.split('\n'))
        print(f"📝 Generated {question_count} questions")
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        sys.exit(1)
    
    # ------------------------------------------------------------
    #   Generate answers
    # ------------------------------------------------------------
    
    print("🔍 Creating vector store...")
    try:
        vector_store = create_vector_store(docs_a, "gemini")
        print("✅ Vector store created successfully")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        sys.exit(1)
    
    print("💡 Generating answers...")
    try:
        retrieve_answers(
            questions=questions_text,
            vector_store=vector_store, 
            provider="gemini"
        )
        print("✅ Answers generated successfully")
        print("📁 Results saved to outputs/answers.txt")
    except Exception as e:
        print(f"❌ Error generating answers: {e}")
        sys.exit(1)
    
    print("🎉 Study Buddy completed successfully!")
    print("📊 Check your LangSmith dashboard to view detailed traces")

if __name__ == "__main__":
    main()
