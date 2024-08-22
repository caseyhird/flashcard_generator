from fastapi import FastAPI, File, UploadFile
# from pdfminer.high_level import extract_text
from pdfplumber import PDF
import io

app = FastAPI()

@app.get("/")
def read_root():
    return "This is the root"

@app.post("/uploadpdf/")
async def create_upload_file(file: UploadFile = File(...)):
    print(f"Received file {type(file)}")
    print(f"File size {file.size}")
    print(f"file type {type(file)}")

    pdf_content = await file.read()
    print(f"pdf_content type {type(pdf_content)}")
    pdf = PDF(io.BytesIO(pdf_content))
    print(f"pdf type {type(pdf)}")

    print(f"Number of pages: {len(pdf.pages)}")
    print(f"Metadata: {pdf.metadata}")

    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    print(f"Text preview: {text}")

    return {"filename": file.filename}