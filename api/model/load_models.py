import joblib # pyright: ignore[reportMissingImports]
from pathlib import Path
from scripts.model.tuning import HyperParamSearch
from scripts.data.feature_select import FeatureSelector
from scripts.data.feature_engineer import FeatureEngineer
from api.schema import PropulsionInputBase, PropulsionInputFull
from scripts.config import cnfg

MODELS = {}

MODEL_PATHS = {
    cnfg["models"]["model_names"]["limited_feat"]: \
        cnfg["models"]["model_dir"] / cnfg["models"]["final_model_file"],
    cnfg["models"]["model_names"]["full_feat"]: \
        cnfg["models"]["model_dir_all_feats"] / cnfg["models"]["final_model_file"],
    }

PREPROCESSOR_PATHS = {
    cnfg["models"]["model_names"]["limited_feat"]: \
         cnfg["models"]["model_dir"] / cnfg["data"]["data_preprocessing"]["preprocessor_file"],
    cnfg["models"]["model_names"]["full_feat"]: \
         cnfg["models"]["model_dir_all_feats"] / cnfg["data"]["data_preprocessing"]["preprocessor_file"],
    }


def load_models(model_path=MODEL_PATHS,
                preprocessor_path=PREPROCESSOR_PATHS,
                loaded_models=MODELS):
    """
    Load models and preprocessor pipeline from given paths into a dictionary,
        skipping missing files.

    :param model_path: Dict of model names to file paths. Defaults to MODEL_PATHS.
    :param preprocessor_path: Dict of model names and paths to respective 
        preprocessor pipelines. Defaults to PREPROCESSOR_PATH.
    :param loaded_models: Dict to store loaded models. Defaults to MODELS.
    :return: Nested dictionary with model name, 
        respective "model" and "preprocessor" files. 
    """
    for name, path in model_path.items():
        if path.exists():
            loaded_models[name] = {"model":joblib.load(path)}
            print(f"loaded {name} model.")
        else:
            print(f"Model {name} not found, continuing w/o it.")

    for name, path in preprocessor_path.items():
        if path.exists():
            loaded_models[name]["preprocessor"] = joblib.load(path)
            print(f"loaded {name} preprocessor.")
        else:
            print(f"Preprocessor {name} not found, continuing w/o it.")


def choose_model(input_data,
                 input_schemas:list=None,
                 model_names:list[str]=None)->str:
    """
    Select the appropriate model by matching input data keys to predefined input schemas.

    :param input_data: Dictionary or JSON of input feature names and values.
    :param input_schemas: Optional list of Pydantic schema classes for each model.
                          Defaults to [PropulsionInputBase, PropulsionInputFull].
    :param model_names: Optional list of model names matching the schemas.
                        Defaults to keys from MODEL_PATHS.
    :return: Name of the matching model, or an error message if no match is found.
    """
    input_schemas = input_schemas or [PropulsionInputBase, PropulsionInputFull]
    model_names = model_names or list(MODEL_PATHS.keys())
    if len(model_names) == len(input_schemas):
        feature_names = {key: input_schemas[index] for index, key in enumerate(model_names)}
    else:
        return "Mismatch between model schema and model type counts."
    for model_type, schema in feature_names.items():
        svchema_keys = schema.model_fields.keys()

        if set(input_data.keys()) == set(svchema_keys):
            print(model_type)  # remove after testing
            print(input_data)  # remove after testing
            return model_type
        
    return "Feature names or count mismatch fields in input schemas for 'full feature set' predictions."

