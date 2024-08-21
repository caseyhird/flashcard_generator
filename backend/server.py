from fastapi import FastAPI, File, UploadFile
# from pdfminer.high_level import extract_text

app = FastAPI()

@app.get("/")
def read_root():
    return "This is the root"

@app.post("/uploadpdf/")
async def create_upload_file(file: UploadFile = File(...)):
    print(f"Received file {type(file)}")
    print(f"File size {file.size}")
    return {"filename": file.filename}

    contents = await file.read()
    decoded_contents = contents.decode('utf-8')
    print(f"file {type(file)}")
    print(f"contents {type(contents)}")
    #text = extract_text(file)   
    print(f"decoded_contents {type(decoded_contents)}")

    return {"filename": file.filename}
