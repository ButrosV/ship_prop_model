from typing import List
from sqlalchemy import desc # pyright: ignore[reportMissingImports]
from sqlalchemy.orm import Session # pyright: ignore[reportMissingImports]
from fastapi import APIRouter, HTTPException, status, Depends # pyright: ignore[reportMissingImports]
from api.schema import PropulsionOutput, PropulsionInputFull
from scripts.model.get_model import load_train_model
from api.model.load_models import load_models, MODELS, choose_model

# from ..oauth2 import get_current_user


router = APIRouter(
    prefix = "/predict",  # add prefix to each router
    tags=["predict"]  # add tag to each router, no need to add individual tags to each router
)

@router.post("/", response_model=PropulsionOutput, status_code=status.HTTP_201_CREATED)
def predict(input_data: PropulsionInputFull):
    """path/endpoint operation function"""
    # Load model and preprocessor based on choose_model
    model_type = choose_model(input_data=input_data)
    the_model, the_preprocessor = MODELS[model_type]["model"], MODELS[model_type]["preprocessor"]
    result = the_model.predict(the_preprocessor.transform(input_data))
    return result
