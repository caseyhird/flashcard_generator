from flashcards_model import FlashCard
from typing import List
import os
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import time
import logging
import json

CHAT_PROMPT_TEMPLATE = """Here is a chunk of text from some notes that I took: 

{text}

Give me 1-3 flashcards for these notes. Keep each question to no more than ~20 words. Include a source or sources quoted directly from
the text which you are using as justification for the answer on the flashcard.
Return your answer in json format as a list of flashcards with question, anwer, and sources. Like this:

{json_example}

I want to parse your response as json, so don't include anything else in your response.
Also make sure that your response contains only valid json, e.g. avoid invalid escape characters.
"""
JSON_EXAMPLE = """
[
    {
        "question": str,
        "sources": List[str],
        "answer": str
    }
]
"""

def _gen_create_flashcards(
        text: str,
        llm: ChatOpenAI,
        max_concurrency: int,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[FlashCard]:
    text_chunks = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_text(text)
    logging.info(f"Generating flashcards from from {len(text_chunks)} chunks")

    prompt_template = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)     
    chain = prompt_template | llm

    inputs = [{"json_example": JSON_EXAMPLE, "text": text} for text in text_chunks]
    try:
        response = chain.batch(inputs, config={"max_concurrency": max_concurrency})
    except Exception as e:
        logging.error(f"Error in model: {e}")
        raise e

    flashcards = []
    for r in response:
        # TODO: figure out json parsing errors
        try:
            for card in json.loads(r.content):
                flashcards.append(FlashCard(card['question'], card['sources'], card['answer']))
        except Exception as e:
            logging.error(f"Error parsing flashcards from model response: {e}")
    

    logging.info(f"Generated {len(flashcards)} flashcards from {len(text_chunks)} chunks")
    return flashcards

def create_flashcards_model_v2(
        text: str,
    ) -> List[FlashCard]:
    max_concurrency = int(os.environ["V2_MAX_CONCURRENCY"])
    chunk_size = int(os.environ["V2_CHUNK_SIZE"])
    chunk_overlap = int(os.environ["V2_CHUNK_OVERLAP"])

    # TODO add error logging
    llm = ChatOpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=os.environ["TOGETHER_API_KEY"],
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )
    
    start_time = time.time()
    flashcards = _gen_create_flashcards(text, llm, max_concurrency, chunk_size, chunk_overlap)
    logging.info(f"Generated cards in {time.time() - start_time:.3f} seconds")
    
    return flashcards