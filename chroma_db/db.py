from typing import Dict, Union, List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from chromadb import PersistentClient
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()

PERSIST_DIR = os.getenv("PERSIST_DIR")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


class ChromaDBManager:
    """
    Manages ChromaDB collections, vector storage, and retrievers
    using the NEW Chroma architecture.
    """

    def __init__(self):
        # NEW Chroma Client (no migration warnings)
        self.client = PersistentClient(path=PERSIST_DIR)

        # Embedding model
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    # -------------------------------------------------------------
    # CREATE COLLECTION FROM DOCUMENT CHUNKS
    # -------------------------------------------------------------
    def create_chroma_collection(self, chunks: List[str], collection_name: str) -> Chroma:
        """
        Create a Chroma vector store collection from text chunks.
        """

        documents = [Document(page_content=chunk) for chunk in chunks]

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            collection_name=collection_name,
            client=self.client
        )

        return vectorstore

    # -------------------------------------------------------------
    # GET OR CREATE COLLECTION
    # -------------------------------------------------------------
    def get_or_create_collection(self, collection_name: str) -> Chroma:
        """
        Load existing collection or create a new one.
        """

        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            client=self.client
        )

    # -------------------------------------------------------------
    # GET SINGLE RETRIEVER
    # -------------------------------------------------------------
    def get_retriever(self, collection_name: str) -> BaseRetriever:
        """
        Get retriever for a specific collection.
        """

        vectorstore = self.get_or_create_collection(collection_name)
        return vectorstore.as_retriever()

    # -------------------------------------------------------------
    # LOAD ALL COLLECTIONS & RETRIEVERS
    # -------------------------------------------------------------
    def get_all_retrievers(self) -> Dict[str, Dict[str, Union[Chroma, BaseRetriever]]]:
        """
        Load all collections and return vectorstore + retriever.
        """

        results = {}

        for c in self.client.list_collections():
            vectorstore = Chroma(
                collection_name=c.name,
                embedding_function=self.embedding_function,
                client=self.client
            )
            results[c.name] = {
                "vectorstore": vectorstore,
                "retriever": vectorstore.as_retriever()
            }

        return results

    # -------------------------------------------------------------
    # LIST ALL COLLECTIONS
    # -------------------------------------------------------------
    def list_collections(self):
        return [c.name for c in self.client.list_collections()]
