import pandas as pd # pyright: ignore[reportMissingModuleSource]
from fastapi import HTTPException  # pyright: ignore[reportMissingImports]


def check_df(input_data):
    """
    Validate and convert input data to a pandas DataFrame if necessary.
    :param input_data: Input data to validate and convert. Expected types are 
        `dict` (or JSON-like structure) or `pd.DataFrame`.
    :return: A pandas DataFrame constructed from the input data.
    :raises HTTPException: If input is neither a dictionary nor a DataFrame.
    """
    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        return input_data
    else:
        message = "Cannot work with provided input: has to be dictionary/json or DataFrame"
        print(message)
        raise HTTPException(
            status_code=418,  # I'm a teapot
            detail=message
        )
    

def organize_input(input_data:pd.DataFrame, preprocessor):
    """
    Align and validate input DataFrame columns to match the expected model input features.
    1) Compare the columns of the input DataFrame with the feature names expected by the fitted
        preprocessor. 
    2) For limited "task_specific_limited" input feature set reorder features and fill missing
        ones with a string "dummy value".
    :param input_data: A pandas DataFrame containing input features for prediction.
    :param preprocessor: A fitted sklearn-compatible object (e.g., Pipeline)
                         that exposes the expected input features via the `feature_names_in_` attribute,
                         either directly or within its first step.
    :return: A DataFrame with columns aligned to the model's expected input features.
    :raises HTTPException: If input columns do not match expected features.
    """
    input_cols = input_data.columns.to_list()
    if hasattr(preprocessor, "feature_names_in_"):
        preprocesoor_input_specs = preprocessor.feature_names_in_
    elif hasattr(preprocessor, "steps"):  # probably a Pipeline
        preprocesoor_input_specs = preprocessor.steps[0][1].feature_names_in_
    else:
        raise AttributeError("Preprocessor does not contain `feature_names_in_`.")

    if input_cols == preprocesoor_input_specs:
        return input_data
    
    elif set(input_cols) == set(preprocesoor_input_specs):
        return input_data[preprocesoor_input_specs]
    
    elif set(input_cols) < set(preprocesoor_input_specs):
        return input_data.reindex(columns=preprocesoor_input_specs)\
            .fillna("dummy value")  # dummy value for columns that will be removed. if not removed by \
        # "task_specific_limited" preprocessor will rise error as numeric values expected for \
        # "full feature set".
    
    else:
        message = f"Cannot work with provided input: mismatch between\
            input columns and inputs model was trained on. Expected input features:\
                \n{preprocesoor_input_specs}"
        print(message)
        raise HTTPException(
            status_code=418,  # I'm a teapot
            detail=message
        )
    