from fastapi import Depends, FastAPI, HTTPException, Header, UploadFile, File, status
from fastapi.responses import FileResponse
import pytesseract
import pathlib, io, uuid, os
from PIL import Image
from .config import Settings, get_settings

tags_metadata = [
    {
        "name": "APP",
        "description": "The main app for extracting text from images.",
    },
    {
        "name": "Development",
        "description": "These urls are only for development purposes.",
    },
]

app = FastAPI(
    title="Extract text API",
    description="A Microservice to extract text from Images using tesseract OCR",
    version="0.0.1",
    openapi_tags=tags_metadata,
    )

settings = get_settings()
BASE_DIR = pathlib.Path(__file__).parent
UPLOAD_DIR = pathlib.Path.joinpath(BASE_DIR, "upload")
DEBUG = settings.DEBUG

@app.get("/", tags=["APP"], summary="Home View")
def index():
    """
    Home View of the API
    """
    return {"data": {
        "details": "Extract text from images",
        "endpoint": "/",
        "method": "POST",
        "required": {
            "file": "Image File for extracting text",
            "authorization_token": "Required header with the authorization-token field set to the valid token",
        },
        "documentation_endpoint": "/docs",
    }}

def verify_auth(authorization: Header(None), settings: Settings = Depends(get_settings)):
    if settings.SKIP_AUTH and settings.DEBUG:
        return
    if authorization is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization")

    try:
        label, token = authorization.split()
    except:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Either label or Token is missing")

    if token != settings.AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization")


@app.post("/", tags=["APP"], summary="Extract text from images")
async def make_predictions(file: UploadFile = File(...), authorization_token = Header(None), settings: Settings = Depends(get_settings)):
    """
    This is the Endpoint to extract text from an image, provide the following Arguments
    - **authorization-token**: This field must be include in the header with the valid token that can be find here https://github.com/roushanagarwalla/extract-text-ms
    - **File**: A valid image file from which we want to extract text
    """
    verify_auth(authorization_token, settings)
    bytes_str = io.BytesIO(await file.read())
    try:
        img = Image.open(bytes_str)
    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Image")
    pred = pytesseract.image_to_string(img)
    predictions = [x for x in pred.strip().split("\n")]
    return {"data": predictions, "original": pred}


@app.post("/image-echo", response_class=FileResponse, tags=["Development"], summary="Use to test Upload of file")
async def image_echo_view(file: UploadFile = File(...), settings: Settings = Depends(get_settings)):
    """
        Only work under development mode not for production
    """
    if not os.path.exists(UPLOAD_DIR):
        os.mkdir(UPLOAD_DIR)
    if not settings.ECHO_ACTIVE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Endpoint")
    bytes_str = io.BytesIO(await file.read())
    try:
        img = Image.open(bytes_str)
    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Image")

    fname = pathlib.Path(file.filename)
    fext = fname.suffix
    dest = UPLOAD_DIR / f"{uuid.uuid1()}{fext}"
    img.save(dest)
    return dest
