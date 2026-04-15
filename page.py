from typing import Annotated
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

filepath=r"C:\Users\Admin\Desktop\Python_api/page.py"

@app.post("/files")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    return {"filename": "file.filename"}
