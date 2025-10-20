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
from src.prompts import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS

# Ensure .env is loaded so API keys are available to providers
load_dotenv()

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
    """Create and return a refine-based summarize chain for question generation."""
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
    """Run the question-generation chain on `docs` and return the questions text."""
    chain = get_question_chain(provider=provider, model=model, temperature=temperature)
    return chain.run(docs)
