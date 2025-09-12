import joblib # pyright: ignore[reportMissingImports]
from pathlib import Path
from scripts.model.tuning import HyperParamSearch
from scripts.data.feature_select import FeatureSelector
from scripts.data.feature_engineer import FeatureEngineer
from scripts.config import cnfg

MODELS = {}

MODEL_PATHS = {
    "task_specific_limited": cnfg["models"]["model_dir"] / cnfg["models"]["final_model_file"],
    # "full_feature_set": cnfg["models"]["model_dir_all_feats"] / cnfg["models"]["final_model_file"],
    "full_feature_set": Path("C:\\Users\\Peteris\\Dropbox\\aktuali_cv\\2025\\shiprojects\\hw_app\\models\\all_model_compar_xgb_model.pkl")  # remove after testing
    }

def load_models(model_path=MODEL_PATHS, loaded_models=MODELS):
    """
    Load models from given paths into a dictionary, skipping missing files.

    :param model_path: Dict of model names to file paths. Defaults to MODEL_PATHS.
    :param loaded_models: Dict to store loaded models. Defaults to MODELS.
    """
    for name, path in model_path.items():
        if path.exists():
            loaded_models[name] = joblib.load(path)
            print(f"loaded {name} model.")
        else:
            print(f"Model {name} not found, continuing w/o it.")
