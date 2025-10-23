# Study Buddy 🧠

Study Buddy is an intelligent study assistant that automatically generates practice questions and answers from your study materials using advanced AI models. Built with LangChain, it supports multiple LLM providers and creates comprehensive study resources from PDF documents.

## 🚀 Features

### Core Functionality
- **Automatic Question Generation**: Creates exam-focused questions from study materials
- **Intelligent Answer Generation**: Provides detailed answers using retrieval-augmented generation
- **Multi-Provider Support**: Works with OpenAI GPT and Google Gemini models
- **Smart Document Processing**: Handles large PDFs with intelligent chunking strategies

### Technical Features
- **LangChain Integration**: Advanced prompt chaining and document processing
- **Vector Database**: ChromaDB for efficient similarity search and retrieval
- **Token-Aware Splitting**: Optimized document chunking for different use cases
- **Configurable Models**: Support for various embedding and language models
- **Comprehensive Documentation**: Well-documented codebase with detailed docstrings

## 📁 Project Structure

```
study_buddy/
├── src/                          # Main source code
│   ├── __init__.py               # Package initialization
│   ├── load_and_split.py         # Document loading and chunking
│   ├── generating_questions.py   # Question generation pipeline
│   ├── generating_answers.py     # Answer generation pipeline
│   └── prompts.py                # Prompt templates
├── study_material/               # Input PDF files
├── outputs/                      # Generated answers and results
├── showcase.ipynb                # Interactive demonstration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd study_buddy
```

### 2. Create Virtual Environment
```bash
python3 -m venv study_buddy_venv
source study_buddy_venv/bin/activate  # On Windows: study_buddy_venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the project root:
```bash
# API Keys (choose one or both)
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# File path for processing
FILE_PATH=study_material/your_document.pdf
```

## 🚀 Quick Start

### Using the Jupyter Notebook
1. Start Jupyter: `jupyter notebook`
2. Open `showcase.ipynb`
3. Run the cells to see the complete workflow

### Using Python Scripts
```python
from src.load_and_split import load_and_split
from src.generating_questions import generate_questions
from src.generating_answers import create_vector_store, retrieve_answers

# Load and split your document
docs_q, docs_a = load_and_split("study_material/your_document.pdf")

# Generate questions
questions = generate_questions(docs_q, provider="gemini")

# Create vector store and generate answers
vector_store = create_vector_store(docs_a, "openai")
retrieve_answers(questions, vector_store, provider="gemini")
```

## 📚 Usage Examples

### Question Generation
```python
from src.generating_questions import generate_questions

# Generate questions with different providers
questions_openai = generate_questions(docs, provider="openai", temperature=0.3)
questions_gemini = generate_questions(docs, provider="gemini", temperature=0.4)
```

### Answer Generation
```python
from src.generating_answers import create_vector_store, get_answer_chain

# Create vector store
vector_store = create_vector_store(docs, "openai")

# Get answer chain
chain = get_answer_chain(vector_store, provider="gemini")

# Ask questions
answer = chain.run("What is machine learning?")
```

## ⚙️ Configuration

### Model Settings
- **OpenAI Models**: `gpt-3.5-turbo-16k` (default), `gpt-4`, etc.
- **Gemini Models**: `gemini-2.0-flash` (default), `gemini-pro`, etc.
- **Embedding Models**: Configurable per provider

### Chunking Parameters
- **Question Generation**: Large chunks (10,000 tokens) for comprehensive context
- **Answer Generation**: Smaller chunks (1,000 tokens) for precise retrieval

## 🔧 Advanced Usage

### Custom Prompts
Modify `src/prompts.py` to customize question and answer generation:
```python
# Custom question prompt
CUSTOM_QUESTION_PROMPT = """
Your custom prompt here...
"""
```

### Batch Processing
```python
# Process multiple documents
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
for doc in documents:
    docs_q, docs_a = load_and_split(doc)
    questions = generate_questions(docs_q)
    # Process questions...
```

## 📊 Output

The system generates:
- **Numbered Questions**: Formatted study questions (e.g., "1. What is...?")
- **Detailed Answers**: Comprehensive responses saved to `outputs/answers.txt`
- **Console Output**: Real-time progress and results

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Studying! 📚✨**