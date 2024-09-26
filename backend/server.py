from fastapi import FastAPI, File, UploadFile
from pdfplumber import PDF
import io
from backend.flashcards_model import create_flashcards_model, FlashCard
from backend.flashcards_model_v2 import create_flashcards_model_v2
from typing import List
import logging
import sys
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

app = FastAPI()

origins = [
    "http://localhost:5173",
    "https://flashcard-frontend-rosy.vercel.app",
    "https://flashcard-frontend-git-main-sriram-hathwars-projects.vercel.app",
    "https://flashcard-frontend-imowb96v0-sriram-hathwars-projects.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return "This is the root"

@app.post("/create-flashcards/")
@app.post("/uploadpdf/")
async def create_flashcards(file: UploadFile = File(...)) -> List[FlashCard]:
    logging.info(f"Running create_flashcards")
    logging.info(f"Received file of size {file.size}")

    pdf_content = await file.read()
    pdf = PDF(io.BytesIO(pdf_content))
    text = "".join([page.extract_text() for page in pdf.pages])

    logging.info(f"Extracted text of length {len(text)}")
    logging.info(f"Generating flashcards ...")
    try:
        flashcards = create_flashcards_model(text)
        logging.info(f"Generated {len(flashcards)} flashcards")
    except Exception as e:
        logging.error(f"Error generating flashcards: {e}")
        raise e
    return flashcards

@app.post("/create-flashcards-v2/")
async def create_flashcards_v2(file: UploadFile = File(...)) -> List[FlashCard]:
    logging.info(f"Running create_flashcards_v2")
    logging.info(f"Received file of size {file.size}")

    pdf_content = await file.read()
    pdf = PDF(io.BytesIO(pdf_content))
    text = "".join([page.extract_text() for page in pdf.pages])

    logging.info(f"Extracted text of length {len(text)}")
    logging.info(f"Generating flashcards ...")
    try:
        flashcards = create_flashcards_v2_model(text)
        logging.info(f"Generated {len(flashcards)} flashcards")
    except Exception as e:
        logging.error(f"Error generating flashcards: {e}")
        raise e
    return flashcards

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
