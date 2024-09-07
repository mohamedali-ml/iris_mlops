from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Replace with your experiment ID
EXPERIMENT_ID = "0"

# Function to get the latest run ID
def get_latest_run_id():
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[EXPERIMENT_ID], filter_string="",
                              run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
                              max_results=1, order_by=["start_time desc"])
    if runs:
        return runs[0].info.run_id
    raise ValueError("No runs found in the specified experiment.")

# Load the model
def load_model():
    latest_run_id = get_latest_run_id()
    model_uri = f"runs:/{latest_run_id}/random_forest_model"
    return mlflow.pyfunc.load_model(model_uri)

model = load_model()

# Define request body
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict/")
def predict(request: IrisRequest):
    data = pd.DataFrame([[request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]],
                        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
