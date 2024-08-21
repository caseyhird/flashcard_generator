from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores.base import VectorStore
from langchain_core.embeddings.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

class FlashCardGenerator:
    def __init__(self):
        pass

    def generate_flashcards(self, ) -> List[str]:
        pass

