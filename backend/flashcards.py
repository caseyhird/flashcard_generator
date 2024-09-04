from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores.base import VectorStore
from langchain_core.embeddings.embeddings import Embeddings
from langchain_together.embeddings import TogetherEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
import os
import time
import logging
from langchain_core.runnables import RunnablePassthrough

QUESTION_PROMPT = """Here are some notes that I took: {pdf_content}. 
Give me 1-3 flashcard questions for these notes. Keep each question to no more than ~20 words and return only these questions divided by newline characters.
Don't add any sort of numbering or bullet points with * or - to the questions, just have the text of the questions and the separating newline characters.
Don't say anything else.  """
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

def calculate_time(func):
    def wrapper(*args, **kwargs):
        begin = time.time()        
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"Finished {func.__name__} in {end - begin:.3f} seconds")
        return result

    return wrapper


# (1) PDF --> string
'''
This is already done in the server.py step when the user uploads the file.  We opted not to copy this function from Google colab.
'''

# (2) chunk pdf text and apply prompt
@calculate_time
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
    pdf_chunk_size = 2000
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=pdf_chunk_size, chunk_overlap=chunk_overlap) 

    return text_splitter.split_text(full_text)

# (3) list of questions results from passing this to the LLM
@calculate_time
def gen_questions(
        llm: BaseChatModel, 
        pdf_content_chunks: List[str]
    ) -> List[str]:
    """Uses the given the LM & prompts to generate study questions"""
    # TODO: handle extracting multiple questions per chunk
    prompt_template = PromptTemplate.from_template(QUESTION_PROMPT) 
    chain = prompt_template | llm
    # TODO: look into concurrency of langchain llm invocations
    inputs = [{"pdf_content": chunk} for chunk in pdf_content_chunks]
    response = chain.batch(inputs)

    questions = []
    for r in response:
        qs = [q.strip() for q in r.content.split('\n')]
        questions += qs
    return questions

# (4) create and populate vector DB
@calculate_time
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

def gen_create_flashcard(llm: BaseChatModel, question: str, vector_store: VectorStore) -> FlashCard:
    def format_sources_list(sources_list):
        return [s.page_content for s in sources_list]

    def format_prompt_input(input):
        return "\n\n".join(input["sources_list"])

    def format_answer(llm_response):
        return llm_response.content
    
    # TODO: allow for more sources, store keeps returning copies of the same result
    retriever = vector_store.as_retriever(search_kwargs={'k': 1})
    prompt_template = PromptTemplate.from_template(ANSWER_PROMPT)
    chain = RunnablePassthrough.assign(
        sources_list=(retriever | format_sources_list)
    ).assign(
        sources=format_prompt_input
    ).assign(
        answer=(prompt_template | llm | format_answer)
    )
    response = chain.invoke({"question": question})
    return FlashCard(question, response["sources_list"], response["answer"])


def generate_flashcards(
        text: str
    ) -> List[FlashCard]:
    # TODO add error logging
    llm = ChatOpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=os.environ["TOGETHER_API_KEY"],
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )

    pdf_content_chunks = chunk_pdf_for_questions_prompt(text) 
    questions = gen_questions(llm, pdf_content_chunks)
    vector_store = gen_vector_store(text, TogetherEmbeddings())
    
    flashcards = []
    start_time = time.time()
    for question in questions: 
        flashcard = gen_create_flashcard(llm, question, vector_store)
        flashcards.append(flashcard)
    logging.info(f"Got answers and sources in {time.time() - start_time:.3f} seconds")
    
    return flashcards
