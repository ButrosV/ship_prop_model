from fastapi import FastAPI, status, Response # pyright: ignore[reportMissingImports]
from api.schema import PropulsionInputBase, PropulsionInputFull, PropulsionOutput
from scripts.model.get_model import load_train_model

app = FastAPI()

@app.post("/predict", response_model=PropulsionOutput, status_code=status.HTTP_201_CREATED)
def predict(input_data: PropulsionInputBase):
    """path/endpoint operation function"""
    the_model, the_preprocessor = load_train_model()
    result = the_model.predict(the_preprocessor.transform(input_data))
    return result
