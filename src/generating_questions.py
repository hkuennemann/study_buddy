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
- The refine chain uses distinct prompts for initial and refine steps.
"""

from typing import List
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from src.prompts import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS

# Hyperparameters
TEMPERATURE = 0.4
MODEL = "gpt-3.5-turbo-16k"

# -----------------------------------------------------------------------------
#   Summarization chain for question generation
# -----------------------------------------------------------------------------
def get_question_chain(model: str = MODEL, temperature: float = TEMPERATURE):
    """Create and return a refine-based summarize chain for question generation."""
    llm = ChatOpenAI(model=model, temperature=temperature)
    return load_summarize_chain(
        llm=llm,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS,
    )

def generate_questions(
    docs: List[Document],
    model: str = MODEL,
    temperature: float = TEMPERATURE,
) -> str:
    """Run the question-generation chain on `docs` and return the questions text."""
    chain = get_question_chain(model=model, temperature=temperature)
    return chain.run(docs)
