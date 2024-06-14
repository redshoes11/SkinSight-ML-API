from fastapi import FastAPI, File, UploadFile
import uvicorn

from image_services import predict, read_image
from utils.format_output import ImagePred

app = FastAPI()

@app.get('/')
def hello_world():
    return {'message': 'API Running'}

@app.post('/api/predict', response_model=ImagePred)
async def predict_image(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    # Read the image
    image = read_image(await file.read())
    # Predict and format
    prediction = predict(image)
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='localhost')