"""
Answer generation module for Study Buddy.

This module provides functionality for creating vector stores from documents,
generating answer chains, and retrieving answers to questions using various
LLM providers (OpenAI and Google Gemini). It supports both embedding creation
and question-answering workflows with configurable models and parameters.

Key Functions:
    - create_vector_store: Creates Chroma vector store from documents
    - get_answer_chain: Creates RetrievalQA chain for answering questions
    - retrieve_answers: Processes questions and saves answers to file

Dependencies:
    - langchain_openai: For OpenAI embeddings and chat models
    - langchain_google_genai: For Google Gemini embeddings and chat models
    - langchain_community: For Chroma vector store and RetrievalQA
    - dotenv: For environment variable management

Example:
    >>> from src.generating_answers import create_vector_store, get_answer_chain
    >>> vector_store = create_vector_store(docs, "openai")
    >>> chain = get_answer_chain(vector_store, provider="gemini")
    >>> answer = chain.run("What is machine learning?")
"""
import re
import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.tracers import LangChainTracer
from langsmith import Client


# Ensure .env is loaded so API keys are available to providers
load_dotenv()

# Initialize LangSmith client for tracking
langsmith_client = Client()

# Hyperparameters
TEMPERATURE = 0.1
OPENAI_EMBEDDING_MODEL = ""
OPENAI_MODEL = "gpt-3.5-turbo-16k"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
GEMINI_MODEL = "gemini-2.0-flash"

def create_vector_store(
        docs: List[Document],
        provider: str,
        model: str = None
        ) -> Chroma:
    """
    Create a vector database (Chroma) for answer generation from documents.
    
    This function creates a Chroma vector store by embedding the provided documents
    using the specified embedding model. The vector store can then be used for
    similarity search and retrieval-augmented generation.

    Args:
        docs (List[Document]): A list of LangChain Document objects to embed and store.
            Each document should contain text content that will be converted to embeddings.
        provider (str): The embedding provider to use. Either "openai" or "gemini".
        model (str, optional): Specific embedding model to use. If None, uses the default
            embedding model for the chosen provider. Defaults to None.

    Returns:
        Chroma: A vector store object ready for similarity search and retrieval.
        
    Raises:
        ValueError: If the provider is not "openai" or "gemini".
        
    Note:
        - Requires appropriate API keys to be set in environment variables:
          OPENAI_API_KEY for OpenAI provider, GEMINI_API_KEY for Gemini provider.
        - The embedding process may take time depending on the number and size of documents.
        
    Example:
        >>> docs = [Document(page_content="Machine learning is...")]
        >>> vector_store = create_vector_store(docs, "openai")
        >>> results = vector_store.similarity_search("What is ML?")
    """
    # Create embeddings object
    if provider == "openai":
        model = model or OPENAI_EMBEDDING_MODEL
        embeddings = OpenAIEmbeddings(
                        model=model,
                        openai_api_key=os.getenv("OPENAI_API_KEY")
                        )
    elif provider == "gemini":
        model = model or GEMINI_EMBEDDING_MODEL
        embeddings = GoogleGenerativeAIEmbeddings(
                        model=model,
                        google_api_key=os.getenv("GEMINI_API_KEY")
                        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'gemini'.")

    # Initialize vector store from documents
    vector_store = Chroma.from_documents(docs, embeddings)

    return vector_store

def get_answer_chain(
        vector_store: Chroma,
        provider: str = "openai",
        model: str = None,
        temperature: float = TEMPERATURE
        ):
    """
    Create a RetrievalQA chain for generating answers from a vector store.
    
    This function initializes a Large Language Model and creates a retrieval-based
    question-answering chain that can answer questions using the provided vector store
    as a knowledge base.
    
    Args:
        vector_store (Chroma): A Chroma vector store containing embedded documents
            to be used as the knowledge base for answering questions.
        provider (str, optional): The LLM provider to use. Either "openai" or "gemini".
            Defaults to "openai".
        model (str, optional): Specific model to use. If None, uses the default model
            for the chosen provider. Defaults to None.
        temperature (float, optional): Controls randomness in the model's responses.
            Lower values make responses more deterministic. Defaults to TEMPERATURE (0.1).
    
    Returns:
        RetrievalQA: A configured retrieval chain ready for question answering.
        
    Raises:
        ValueError: If the provider is not "openai" or "gemini".
        
    Example:
        >>> vector_store = create_vector_store(docs, "openai")
        >>> chain = get_answer_chain(vector_store, provider="gemini")
        >>> answer = chain.run("What is machine learning?")
    """
    # Initialize Large Language Model for answer generation
    if provider == "openai":
        model = model or OPENAI_MODEL
        llm_answer_gen = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "gemini":
        model = model or GEMINI_MODEL
        llm_answer_gen = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'gemini'.")

    # Initialize retrieval chain for answer generation
    answer_gen_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
        )

    return answer_gen_chain


def retrieve_answers(
        questions: str,
        vector_store: Chroma,
        provider: str = "openai",
        model: str = None,
        temperature: float = TEMPERATURE,
        limit: int = None
        ):
    """
    Generate answers for a list of questions and save them to a file.
    
    This function processes a string containing numbered questions, generates answers
    using the provided vector store as a knowledge base, and saves the results to
    "answers.txt". Questions are filtered to only include lines that start with a
    number followed by a dot (e.g., "1. What is...?").
    
    Args:
        questions (str): A string containing numbered questions separated by newlines.
            Only questions starting with a number and dot will be processed.
        vector_store (Chroma): A Chroma vector store containing embedded documents
            to be used as the knowledge base for answering questions.
        provider (str, optional): The LLM provider to use. Either "openai" or "gemini".
            Defaults to "openai".
        model (str, optional): Specific model to use. If None, uses the default model
            for the chosen provider. Defaults to None.
        temperature (float, optional): Controls randomness in the model's responses.
            Lower values make responses more deterministic. Defaults to TEMPERATURE (0.1).
        limit (int, optional): Maximum number of questions to answer. If None, answers all questions.
            Defaults to None.
    
    Returns:
        None: This function doesn't return a value but prints progress and saves
            results to "answers.txt".
        
    Note:
        - Questions are filtered using regex pattern r"^\d+\." to only process
          properly numbered questions.
        - Results are appended to "answers.txt" in the current working directory.
        - Progress is printed to console for each question being processed.
        - If limit is specified, only the first N questions will be answered.
        
    Example:
        >>> questions = "1. What is machine learning?\\n2. How does it work?"
        >>> retrieve_answers(questions, vector_store, provider="gemini", limit=1)
        Question: 1. What is machine learning?
        Answer: Machine learning is a subset of artificial intelligence...
        --------------------------------------------------
    """
    # Split generated questions into a list of questions
    question_list = questions.split("\n")

    # Keep only lines that start with a number and a dot
    question_list = [line.strip() for line in question_list if re.match(r"^\d+\.", line.strip())]
    
    # Apply limit if specified
    if limit is not None:
        question_list = question_list[:limit]
        print(f"\t📝 Limited to first {limit} questions")

    # Get answer chain
    answer_chain = get_answer_chain(vector_store, provider, model, temperature)

    # Answer each question and save to a file
    with open("outputs/answers.txt", "w", encoding="utf-8") as f:
        for i, question in enumerate(question_list):
            print(f"Answering question {i+1} of {len(question_list)}", end="\r")

            # Clean question
            clean_question = re.sub(r'^\d+\.\s*', '', question)
            
            # Run the chain
            answer = answer_chain.run(question)
            
            # Save answer to file
            f.write(f"Question {i+1}: " + clean_question + "\n")
            f.write("Answer: " + answer + "\n")
            f.write("--------------------------------------------------\n\n")
