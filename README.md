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

# Optional: Limit number of questions to answer (e.g., 20 for first 20 questions)
# QUESTION_LIMIT=20
```

## 📚 Usage Examples

### Running the Application
```bash
# Run the main application
python main.py
```

The main.py file provides a complete study buddy experience that automatically processes your study materials and generates practice questions and answers.

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

## 📊 Output

The system generates:
- **Numbered Questions**: Formatted study questions (e.g., "1. What is...?")
- **Detailed Answers**: Comprehensive responses saved to `outputs/answers.txt`
- **Console Output**: Real-time progress and results

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Studying! 📚✨**