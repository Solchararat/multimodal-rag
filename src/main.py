from plant_classifier import PlantClassifier
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import base64
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from contextlib import asynccontextmanager

class ImageRequest(BaseModel):
    image: str

db_path = Path(__file__).parent.parent / "db"
plant_classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = Path(__file__).parent.parent / "db"
    app.state.plant_classifier = PlantClassifier(db_path=db_path)
    yield
    del app.state.plant_classifier

app = FastAPI(lifespan=lifespan)

@app.post("/classify")
async def classify_plant(request: ImageRequest):
    try:

        
        image_bytes = base64.b64decode(request.image)
        image = Image.open(BytesIO(image_bytes))
        
        result, _ = plant_classifier.classify_plant(image)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))