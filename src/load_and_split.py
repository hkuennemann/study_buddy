"""
Load a PDF and produce token-aware chunks for question and answer workflows.

This module provides a single entry point, `load_and_split`, which:
- Loads text from a PDF using LangChain's `PyPDFLoader`
- Creates large chunks for question generation and smaller chunks for answer generation
- Returns two sequences of `Document` objects: `(docs_q, docs_a)`

Parameters (of `load_and_split`):
- file_path: Path to the source PDF.
- model_name: Tokenizer model name used by `TokenTextSplitter` (affects token counts).
- chunk_size_q / chunk_overlap_q: Chunking strategy for question generation.
- chunk_size_a / chunk_overlap_a: Chunking strategy for answer generation.

Returns:
- Tuple[List[Document], List[Document]]: `(docs_q, docs_a)` where:
  - `docs_q` are larger question-oriented chunks
  - `docs_a` are smaller answer-oriented chunks derived from `docs_q`

Example:
    from src.load_and_split import load_and_split
    docs_q, docs_a = load_and_split("study_material/Advanced Data Analytics.pdf")

CLI:
    When executed directly, prints the number of produced question and answer chunks.

Notes:
- The chosen `model_name` only affects tokenization for splitting; it does not call the model.
- Consider attaching source metadata (e.g., page numbers) upstream if you need citations.
- For large PDFs, you may want to cache the resulting chunks to speed up iteration.
"""

import os
import logging
from typing import List, Tuple
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

def load_and_split(
    file_path: str,
    model_name: str = "gpt-3.5-turbo-16k",
    chunk_size_q: int = 10000,
    chunk_overlap_q: int = 200,
    chunk_size_a: int = 1000,
    chunk_overlap_a: int = 100,
    suppress_warnings: bool = False
) -> Tuple[List[Document], List[Document]]:
    """
    Load and split data from a PDF file into chunks for question and answer generation.

    Args:
        file_path: Path to the PDF file
        model_name: Name of the model to use for tokenization
        chunk_size_q: Size of the chunks for question generation
        chunk_overlap_q: Overlap for the chunks for question generation
        chunk_size_a: Size of the chunks for answer generation
        chunk_overlap_a: Overlap for the chunks for answer generation
    
    Returns:
        Tuple[List[Document], List[Document]]: A tuple containing the chunks for question 
            and answer generation
    """

    # Suppress warnings if verbose is False
    if suppress_warnings:
        logging.getLogger("pypdf").setLevel(logging.ERROR)

    # ------------------------------------------------------------
    # Load data from PDF
    # ------------------------------------------------------------

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # --- Note: The data variable holds the extracted study material, 
    # --- neatly organized and ready for further processing

    # ------------------------------------------------------------
    # Split data into chunks for question generation
    # ------------------------------------------------------------

    # Combine text from Document into one string for question generation
    text_question_gen = "".join(page.page_content for page in data)

    # Initialize Text Splitter for question generation
    q_splitter = TokenTextSplitter(
        model_name=model_name, 
        chunk_size=chunk_size_q, 
        chunk_overlap=chunk_overlap_q
        )

    # Split text into chunks for question generation
    text_chunks_q = q_splitter.split_text(text_question_gen)

    # Convert chunks into Documents for question generation
    docs_q = [Document(page_content=t) for t in text_chunks_q]

    # ------------------------------------------------------------
    # Split data into chunks for answer generation
    # ------------------------------------------------------------

    # Initialize Text Splitter for answer generation
    a_splitter = TokenTextSplitter(
        model_name=model_name, 
        chunk_size=chunk_size_a, 
        chunk_overlap=chunk_overlap_a)

    # Split documents into chunks for answer generation
    docs_a = a_splitter.split_documents(docs_q)
    return docs_q, docs_a

if __name__ == "__main__":
    # Simple CLI entry for quick checks
    # Load .env file
    load_dotenv() 

    # Get FILE_PATH
    file_path = os.getenv("FILE_PATH")

    if not file_path:
        raise ValueError("FILE_PATH environment variable not set in .env")

    dq, da = load_and_split(file_path)
    print(f"Question chunks: {len(dq)}, Answer chunks: {len(da)}")
