from fastapi import FastAPI
from schema import PropulsionInput, PropulsionOutput
from scripts.model.get_model import load_train_model

app = FastAPI()

@app.post("/predict", response_model=PropulsionOutput)
def predict(input_data: PropulsionInput):
    the_model, the_preprocessor = load_train_model()
    result = the_model.predict(the_preprocessor.transform(input_data))
    return result
