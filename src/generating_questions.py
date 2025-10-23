"""
Build a LangChain refine chain to generate study questions from documents.

This module exposes two entry points:
- `get_question_chain(model, temperature)` returns a configured summarize/refine chain
- `generate_questions(docs, model, temperature)` runs the chain on input `docs`

Typical usage:
    from src.load_and_split import load_and_split
    from src.generating_questions import generate_questions

    docs_q, _ = load_and_split("study_material/Advanced Data Analytics.pdf")
    questions_text = generate_questions(docs_q)

Notes:
- `ChatOpenAI` reads the `OPENAI_API_KEY` from your environment (.env supported).
- `ChatGoogleGenerativeAI` reads the `GEMINI_API_KEY` from your environment (.env supported).
- The refine chain uses distinct prompts for initial and refine steps.
"""

from typing import List
import os
from dotenv import load_dotenv
from langchain.chains.summarize.chain import load_summarize_chain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tracers import LangChainTracer
from langsmith import Client
from src.prompts import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS

# Ensure .env is loaded so API keys are available to providers
load_dotenv()

# Initialize LangSmith client for tracking
langsmith_client = Client()

# Hyperparameters
TEMPERATURE = 0.4
OPENAI_MODEL = "gpt-3.5-turbo-16k"
GEMINI_MODEL = "gemini-2.0-flash"

# -----------------------------------------------------------------------------
#   Summarization chain for question generation
# -----------------------------------------------------------------------------
def get_question_chain(
    provider: str = "openai",
    model: str = None,
    temperature: float = TEMPERATURE
):
    """
    Create and return a refine-based summarize chain for question generation.
    
    This function initializes a Large Language Model and creates a refine-based
    summarization chain specifically configured for generating study questions
    from documents. The chain uses distinct prompts for initial question generation
    and refinement steps.
    
    Args:
        provider (str, optional): The LLM provider to use. Either "openai" or "gemini".
            Defaults to "openai".
        model (str, optional): Specific model to use. If None, uses the default model
            for the chosen provider. Defaults to None.
        temperature (float, optional): Controls randomness in the model's responses.
            Higher values make responses more creative. Defaults to TEMPERATURE (0.4).
    
    Returns:
        SummarizeChain: A configured refine-based summarization chain ready for
            question generation from documents.
        
    Raises:
        ValueError: If the provider is not "openai" or "gemini".
        
    Note:
        - Requires appropriate API keys to be set in environment variables:
          OPENAI_API_KEY for OpenAI provider, GEMINI_API_KEY for Gemini provider.
        - The chain uses PROMPT_QUESTIONS for initial generation and
          REFINE_PROMPT_QUESTIONS for refinement steps.
        - Verbose mode is enabled by default for debugging purposes.
        
    Example:
        >>> chain = get_question_chain(provider="gemini", temperature=0.3)
        >>> questions = chain.run(documents)
    """
    if provider == "openai":
        model = model or OPENAI_MODEL
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "gemini":
        model = model or GEMINI_MODEL
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'gemini'.")

    return load_summarize_chain(
        llm=llm,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS,
    )

def generate_questions(
    docs: List[Document],
    provider: str = "openai",
    model: str = None,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Generate study questions from a list of documents using a refine-based chain.
    
    This function processes a list of documents and generates comprehensive study
    questions using a refine-based summarization chain. The questions are designed
    to help students prepare for exams by covering the key concepts and information
    present in the source documents.
    
    Args:
        docs (List[Document]): A list of LangChain Document objects containing
            the study material to generate questions from.
        provider (str, optional): The LLM provider to use. Either "openai" or "gemini".
            Defaults to "openai".
        model (str, optional): Specific model to use. If None, uses the default model
            for the chosen provider. Defaults to None.
        temperature (float, optional): Controls randomness in the model's responses.
            Higher values make responses more creative. Defaults to TEMPERATURE (0.4).
    
    Returns:
        str: A string containing the generated study questions, typically formatted
            as numbered questions (e.g., "1. What is...?\\n2. How does...?").
        
    Raises:
        ValueError: If the provider is not "openai" or "gemini".
        
    Note:
        - The function uses a refine-based approach, which is particularly effective
          for processing large documents by generating initial questions and then
          refining them with additional context.
        - The generated questions are designed to be comprehensive and exam-focused.
        - Processing time depends on the number and size of input documents.
        - All operations are tracked in LangSmith for monitoring and debugging.
        
    Example:
        >>> from src.load_and_split import load_and_split
        >>> docs_q, _ = load_and_split("study_material/Advanced Data Analytics.pdf")
        >>> questions = generate_questions(docs_q, provider="gemini")
        >>> print(questions)
        1. What is machine learning?
        2. How does supervised learning work?
        ...
    """
    chain = get_question_chain(provider=provider, model=model, temperature=temperature)
    
    # Run the chain (tracing is automatically enabled via environment variables)
    result = chain.run(docs)
    
    return result
