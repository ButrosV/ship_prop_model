import joblib
from pathlib import Path

from ...scripts.config import cnfg

MODELS = {}

MODEL_PATHS = {
    "task_specific_limited": Path(cnfg["models"]["model_dir"]) / cnfg["models"]["final_model_file"],
    "full_feature_set": Path(cnfg["models"]["model_dir_all_feats"]) / cnfg["models"]["final_model_file"],
    }

def load_models(model_path):
    """Try loading available models, skip missing ones."""
    print(model_path)

load_models(model_path=MODEL_PATHS)
