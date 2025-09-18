from scripts.config import cnfg
from scripts.model.tuning import HyperParamSearch
from scripts.data.feature_select import FeatureSelector
from scripts.data.feature_engineer import FeatureEngineer
from api.model.load_models import load_models, MODELS

# TODO remove file as it is redundant

LIMITED_FEATURES = cnfg["data"]["data_preprocessing"]["task_features"]
ADDITIONAL_FEATURES = cnfg["data"]["data_preprocessing"]["features_to_drop_task"]
FULL_FEATURES = LIMITED_FEATURES + ADDITIONAL_FEATURES

def select_model(input_dict: dict) -> str:
    """Decide whether to use 'limited' or 'full' model.
    return "task_specific_limited", "full_feature_set" or "something went wrong".
    """
    pass

def predict(model_type: str, features: list[float]) -> dict:
    """from `select_model` output string as "model_type" choose model from "MODELS" and
      return {"shaftPower": preds[0], "speedOverGround": preds[1]}"""
    pass
