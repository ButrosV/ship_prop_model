# from typing import List
# import pandas as pd # pyright: ignore[reportMissingModuleSource]
# from sqlalchemy import desc # pyright: ignore[reportMissingImports]
# from sqlalchemy.orm import Session # pyright: ignore[reportMissingImports]
from fastapi import APIRouter, HTTPException, status # pyright: ignore[reportMissingImports], # , Depends
from api.schema import PropulsionOutput, PropulsionInputFull
# from scripts.model.get_model import load_train_model
from api.model.load_models import MODELS, choose_model  # load_models, 
from api.utils.preprocessing import check_df, organize_input


#TODO: update docstrings

router = APIRouter(
    prefix = "/predict",  # add prefix to each router
    tags=["predict"]  # add tag to each router, no need to add individual tags to each router
)

@router.post("/", response_model=PropulsionOutput, status_code=status.HTTP_201_CREATED)


def predict(input_data: PropulsionInputFull):
    """path/endpoint operation function"""
    # Load model and preprocessor based on choose_model
    input_data = input_data.dict(exclude_unset=True)
    model_type = choose_model(input_data=input_data)
    if model_type not in MODELS.keys():
        print(f"Cannot work with provided input: {model_type}")
        raise HTTPException(
            status_code=418,  # I'm a teapot
            detail=model_type
        )
    the_model, the_preprocessor = MODELS[model_type]["model"], MODELS[model_type]["preprocessor"]
    df = check_df(input_data=input_data)
    df = organize_input(df, the_preprocessor)
    predictions = the_model.predict(the_preprocessor.transform(df))
    # TODO consider nicer dynamic, schema based output formatting
    return {"shaftPower": predictions[0][0], "speedOverGround": predictions[0][1]}
