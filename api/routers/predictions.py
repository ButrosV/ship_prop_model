from fastapi import APIRouter, HTTPException # pyright: ignore[reportMissingImports], # , Depends
from api.schema import PropulsionOutput, PropulsionInputFull
from api.model.load_models import MODELS, choose_model
from api.utils.preprocessing import check_df, organize_input


router = APIRouter(
    prefix = "/predict",  # add prefix to each router
    tags=["predict"]  # add tag to each router, no need to add individual tags to each router
)

@router.post("/", response_model=PropulsionOutput, status_code=200)


def predict(input_data: PropulsionInputFull):
    """
    Path/endpoint operation to perform model inference on input data and return predictions.
    1) Select and load from MODELS dictionary the appropriate model type based on the input data.
    2) Validate and format the input data as a DataFrame.
    3) Apply preprocessing and generate model predictions.
    4) Map prediction outputs to PropulsionOutput schema.
    :param input_data: Input features as a Pydantic model (PropulsionInputFull). Automatically parsed 
        from JSON payload of the POST request.
    :return: Dictionary of predicted outputs conforming to the PropulsionOutput schema.
    :raises HTTPException: if the model type cannot be determined or is unsupported or
        if prediction output shape does not match the expected schema.
    """
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

    output_keys = PropulsionOutput.model_fields.keys()
    
    if len(predictions[0]) != len(output_keys):
        message = "API output schema mismatch with prediction output."
        print(message)
        raise HTTPException(
            status_code=418,  # I'm a teapot
            detail=message
        )

    for prediction in predictions:
        return dict(zip(output_keys, prediction))
