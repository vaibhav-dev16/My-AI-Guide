# from langchain.vectorstores import Chroma
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from langchain_community.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
from typing import Union, List, Dict
import chromadb
from chromadb.config import Settings

from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_chroma_collection(chunks: List[str], collection_name: str):
    """
    Creates a Chroma vector store collection from a list of text chunks.

    Args:
        chunks (List[str]): List of text chunks.
        collection_name (str): Name of the Chroma collection to create.

    Returns:
        Chroma: A Chroma vector store object.
    """
    # Initialize sentence transformer embedding model
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # embedding_function = OpenAIEmbeddings()

    # Convert chunks into Document objects (required by LangChain)
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create and return the Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        collection_name=collection_name
    )

    return vectorstore

def get_retriever(collection_name: str = None) -> Dict[str, Dict[str, Union[Chroma, BaseRetriever]]]:
    """
    Get Chroma vectorstore(s) and retriever(s) in a dictionary.

    Args:
        collection_name (str): Name of the Chroma collection to load.  
                               If None, loads all available collections.

    Returns:
        dict: {collection_name: {"vectorstore": Chroma, "retriever": BaseRetriever}}
    """
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    collection_dict = {}

    if collection_name:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function
        )
        collection_dict[collection_name] = {
            "vectorstore": vectorstore,
            "retriever": vectorstore.as_retriever()
        }
        return collection_dict

    # Load all collections
    client = chromadb.Client(Settings())
    all_collections = client.list_collections()

    for collection in all_collections:
        vectorstore = Chroma(
            collection_name=collection.name,
            embedding_function=embedding_function
        )
        collection_dict[collection.name] = {
            "vectorstore": vectorstore,
            "retriever": vectorstore.as_retriever()
        }

    return collection_dict

