from pathlib import Path
import joblib

from xgboost import XGBRegressor
from sklearn.multioutput import RegressorChain

from scripts.config import cnfg
from scripts.model.tuning import HyperParamSearch

def load_train_model(X, y, file_name:str=None, save_folder:Path=None, model=None, model_params:dict=None):
    """
    Train a model with provided parameters or load saved final model.
    If a final model already exists, it is loaded. If not, a best grid search model is attempted to load.
    If neither exists, the model is trained from scratch using provided or default parameters.

    :param X: Training feature data.
    :param y: Training target data.
    :param file_name: Name of the file to save/load the final model. If None, uses default from config.
    :param save_folder: Path to the folder where the model is saved or should be saved.
                        If None, defaults to the configured model directory.
    :param model: Optional custom model to train. If None, defaults to RegressorChain with XGBRegressor.
    :param model_params: Dictionary of model parameters for XGBRegressor. If None, uses best 
                        tuned parameters from config.
    :return: Tuple containing the trained model and the fitted preprocessor.
    """
    folder_path = save_folder or Path(__file__).parent.parent / cnfg["models"]["model_dir"]  # for script
    folder_path.mkdir(parents=True, exist_ok=True)
    file_name = file_name or cnfg["models"]["final_model_file"]
    hyper_tune_file = "grid_search_xgb.pkl" or cnfg["hyperparameter_tuning"]["grid_search_file"]  # remove hardcoded after testing
    final_model_path = folder_path / file_name
    tuned_model_path = folder_path / hyper_tune_file
    preprocessor = HyperParamSearch().preprocess(X=X)

    if model_params:
        final_model = RegressorChain(XGBRegressor(**model_params))
        print("Fitting and saving final model from provided parameters.")  # remove after testing
        X_train_prep = preprocessor.transform(X)
        final_model.fit(X=X_train_prep, Y=y)
    elif model:
        final_model = model
        print("Fitting and saving provided custom model.")  # remove after testing
        X_train_prep = preprocessor.transform(X)
        final_model.fit(X=X_train_prep, Y=y)
    elif final_model_path.is_file():
        final_model = joblib.load(filename=final_model_path)
        print("Loading already saved final model.")  # remove after testing
    elif tuned_model_path.is_file():
        final_model = joblib.load(filename=tuned_model_path)
        print("Loading best grid model.")  # remove after testing
    elif cnfg["models"]["best_tuned_params"]:
        final_model = RegressorChain(XGBRegressor(**cnfg["models"]["best_tuned_params"]))
        print("Fitting and saving final model from config file parameters.")  # remove after testing
        X_train_prep = preprocessor.transform(X)
        final_model.fit(X=X_train_prep, Y=y)
    else:
        print("No parameters to fit final model or saved models found.")
        
    if not final_model_path.is_file():
        joblib.dump(value=final_model, filename=final_model_path)

    return final_model, preprocessor
