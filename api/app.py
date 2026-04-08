from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn

app = FastAPI(title="Sentiment API")

# Load the trained model from MLflow's saved artifact
model = mlflow.sklearn.load_model("mlruns/464371528731484501/aeace55f433b4dd9b13d0940a04e3046/artifacts/model")


class Review(BaseModel):
    text: str

class Result(BaseModel):
    sentiment: str
    confidence: float

@app.post("/predict", response_model=Result)
def predict(review: Review):
    # Get probability scores for class 0 and class 1
    proba = model.predict_proba([review.text])[0]
    label = int(proba.argmax())           # pick whichever is higher
    confidence = float(proba.max())       # how confident (0.0 to 1.0)
    sentiment = "positive" if label == 1 else "negative"

    return Result(sentiment=sentiment, confidence=round(confidence, 4))

@app.get("/health")
def health():
    return {"status": "ok"}

'''
next step : Build Docker Image :- 
    docker build -t sentiment-api .
Run Container:- 
    docker run -p 8000:8000 sentiment-api
then goto :-  http://localhost:8000/docs  ( this is swagger ui )
'''