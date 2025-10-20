from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from typing import List
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import re

# Ensure .env is loaded so API keys are available to providers
load_dotenv()

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

    Args:
        docs (List[Document]): A list of LangChain Document objects to embed.
        provider (str): The provider of your choice. Either ChatGPT or Gemini.
        model (str):
        temperature (float):

    Returns:
        Chroma: A vector store object ready for similarity search.
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
    TODO: Docstring
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
        temperature: float = TEMPERATURE
        ):
    # Split generated questions into a list of questions
    question_list = questions.split("\n")

    # Keep only lines that start with a number and a dot
    question_list = [line.strip() for line in question_list if re.match(r"^\d+\.", line.strip())]

    # Get answer chain
    answer_chain = get_answer_chain(vector_store, provider, model, temperature)

    # Answer each question and save to a file
    with open("answers.txt", "a") as f:
        for question in question_list:
            print("Question: ", question)
            answer = answer_chain.run(question)
            print("Answer: ", answer)
            print("--------------------------------------------------\\n\\n")
            # Save answer to file
            f.write("Question: " + question + "\\n")
            f.write("Answer: " + answer + "\\n")
            f.write("--------------------------------------------------\\n\\n")


