from fastapi import FastAPI, File, UploadFile
from pdfplumber import PDF
import io
from flashcards import generate_flashcards, FlashCard
from typing import List

app = FastAPI()

@app.get("/")
def read_root():
    return "This is the root"

@app.post("/uploadpdf/")
async def create_upload_file(file: UploadFile = File(...)) -> List[FlashCard]:
    # TODO: replace print statements with logging
    print(f"Received file {type(file)}")
    print(f"File size {file.size}")
    print(f"file type {type(file)}")

    pdf_content = await file.read()
    pdf = PDF(io.BytesIO(pdf_content))
    text = "".join([page.extract_text() for page in pdf.pages])

    flashcards = generate_flashcards(text)
    print(f"Generated {len(flashcards)} flashcards")
    return flashcards

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
