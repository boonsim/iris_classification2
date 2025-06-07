from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load the trained model
with open('iris_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the request body
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# For response clarity
iris_species = ['Setosa', 'Versicolor', 'Virginica']

app = FastAPI(title="Iris Classifier API")

@app.post("/predict")
def predict_iris(data: IrisRequest):
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(input_data)[0]
    species = iris_species[prediction]
    return {"prediction": int(prediction), "species": species}