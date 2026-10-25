
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Locate the model from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "machine_failure_model.joblib"

model = joblib.load(MODEL_PATH)
THRESHOLD = 0.4

app = FastAPI(title="Smart Manufacturing Quality Prediction API")


class MachineData(BaseModel):
    Type: str = Field(pattern="^[LMH]$")
    air_temperature: float = Field(alias="Air temperature [K]", gt=0)
    process_temperature: float = Field(alias="Process temperature [K]", gt=0)
    rotational_speed: float = Field(alias="Rotational speed [rpm]", gt=0)
    torque: float = Field(alias="Torque [Nm]", gt=0)
    tool_wear: float = Field(alias="Tool wear [min]", ge=0)


@app.get("/")
def home():
    return {"message": "Smart Manufacturing Quality Prediction API is running"}


@app.post("/predict")
def predict(data: MachineData):
    machine_data = data.model_dump(by_alias=True)

    machine_data["Temperature difference [K]"] = (
        machine_data["Process temperature [K]"]
        - machine_data["Air temperature [K]"]
    )

    input_df = pd.DataFrame([machine_data])
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(probability >= THRESHOLD)

    return {
        "failure_probability": round(probability, 4),
        "threshold": THRESHOLD,
        "prediction": prediction,
        "result": "Failure" if prediction else "No Failure"
    }
