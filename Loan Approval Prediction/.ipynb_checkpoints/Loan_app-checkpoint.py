from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Define the input data model
class InputData(BaseModel):
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float

# Load the scaler and model
scaler = joblib.load("Scaler.pkl")
model = joblib.load("model.pkl")

# Initialize the FastAPI app
app = FastAPI()

@app.post("/predict/")
def predict(input_data: InputData):
    # Prepare the input data for prediction
    x_values = np.array([[
        input_data.x1,
        input_data.x2,
        input_data.x3,
        input_data.x4,
        input_data.x5
    ]])

    # Scale the input data
    scaled_x_values = scaler.transform(x_values)

    # Make a prediction
    prediction = model.predict(scaled_x_values)
    prediction = int(prediction[0])  # Convert prediction to integer if needed

    return {"prediction": prediction}

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
