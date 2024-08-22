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
import os

QUESTION_PROMPT = """Here are some notes that I took: {pdf_content}. 
Give me one study question for these notes. Don't say anything else or add any context/formatting around this question.  """
ANSWER_PROMPT = """Use these sources from my notes to answer a question: {sources}
Here's the question I want to answer. Give me the answer to this question and nothing else. 
Keep your answer to no more than ~20 words. {question}  """
VECTOR_DB_CHUNK_SIZE = 500
# TODO define LLM and embedding model choices

@dataclass
class FlashCard:
    question: str
    sources: List[str]
    answer: str

    # TODO: can delete -- only for debugging
    def __print__(self):
        print("Q: ", self.question)
        print("S: ", self.sources[0])
        print("A: ", self.answer)

# (1) PDF --> string
'''
This is already done in the server.py step when the user uploads the file.  We opted not to copy this function from Google colab.
'''

# (2) chunk pdf text and apply prompt
def chunk_pdf_for_questions_prompt(
        full_text: str,
    ) -> List[str]:
    """
    Prepares the LLM prompt used to generate study questions. Fills in the existing
    prompt template using the given text, dividing this text as needed so the
    filled in prompt will fit the LM context.

    full_text: the entire text of the input PDF
    Returns a list of prompts fully prepared for the LM.
    """
    # NOTE: currently prompt_template_len overestimates num tokens by using num characters
    # TODO set with pdf_chunk_size = lm_context_len - prompt_template_len - num_questions*max_question_len
    pdf_chunk_size = 1500
    chunk_overlap = 200
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=pdf_chunk_size, chunk_overlap=chunk_overlap) 

    return text_splitter.split_text(full_text)

# (3) list of questions results from passing this to the LLM
def gen_questions(
        llm: BaseChatModel, 
        pdf_content_chunks: List[str]
    ) -> List[str]:
    """Uses the given the LM & prompts to generate study questions"""
    # TODO: handle extracting multiple questions per chunk
    prompt_template = PromptTemplate.from_template(QUESTION_PROMPT) 
    chain = prompt_template | llm
    # TODO: look into concurrency of langchain llm invocations
    questions = []
    for chunk in pdf_content_chunks:
        r = chain.invoke({"pdf_content": chunk})
        questions.append(r.content)
    return questions

# (4) create and populate vector DB
def gen_vector_store(
        full_text: str, 
        embedding_model: Embeddings,
        chunk_size: int = VECTOR_DB_CHUNK_SIZE, 
    ) -> VectorStore:
    """
    Creates a vector database and populates it with chunks broken from
    full_text
    """
    # TODO: investigate optimal chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0) 
    chunks = text_splitter.split_documents([Document(full_text)])
    return Chroma.from_documents(chunks, embedding=embedding_model)

# (5) & (6) Get sources for each question from vector store
def gen_sources(question: str, vector_store: VectorStore) -> List[str]:
    # TODO: allow for more sources, store keeps returning copies of the same result
    results = vector_store.similarity_search(question, k=1) 
    return [r.page_content for r in results]

# (7) Use Q & S to get answer from LM
def gen_answer(llm: BaseChatModel, question: str, sources: List[str]) -> str:
    prompt_template = PromptTemplate.from_template(ANSWER_PROMPT)
    chain = prompt_template | llm
    response = chain.invoke({"sources": "\n\n".join(sources), "question": question})
    return response.content

def generate_flashcards(
        text: str
    ) -> List[FlashCard]:
    llm = ChatOpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=os.environ["TOGETHER_API_KEY"],
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )

    pdf_content_chunks = chunk_pdf_for_questions_prompt(text) 
    questions = gen_questions(llm, pdf_content_chunks)
    vector_store = gen_vector_store(text, HuggingFaceEmbeddings())
    
    flashcards = []
    for question in questions: 
        sources = gen_sources(question, vector_store)
        answer = gen_answer(llm, question, sources)
        flashcard = FlashCard(question, sources, answer)
        flashcards.append(flashcard)
    
    return flashcards