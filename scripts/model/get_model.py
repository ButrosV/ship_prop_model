from pathlib import Path
import joblib

from xgboost import XGBRegressor
from sklearn.multioutput import RegressorChain

from scripts.config import cnfg


def _handle_preprocessor(preprocessor_path:Path, X, remove_features:list[str],
                         drop_engineer_source_features:bool):
    """
    Helper function to load exiting preprocessor file or create new one
    :param X: Training feature data.
    :param preprocessor_path: Path to exiting preprocessor location.
    :param remove_features: optional list of features to be removed during preprocessing.
    :param drop_engineer_source_features: handle source features after feature engineering.
    """
    if preprocessor_path.is_file():
        return joblib.load(filename=preprocessor_path)
    elif X is not None and not X.empty:
        from scripts.model.tuning import HyperParamSearch
        return HyperParamSearch(remove_features=remove_features).preprocess(
            X=X, drop_engineer_source_features=drop_engineer_source_features)
    else:
       print("No preprocessor file or training data to configure preprocessing available.")


def _handle_fitting(X, Y, model, preprocessor):
    """
    Helper function to fit model with preprocessed training data.
    :param X: Training feature data.
    :param Y: Training targets.
    :param model: model instance.
    :param preprocessor: Fitted preprocessor.
    :return: Fitted model.
    """
    X_train_prep = preprocessor.transform(X)
    model.fit(X_train_prep, Y)
    return model


def load_train_model(X=None, Y=None, model_file_name:str=None, preprocessor_file_name:str=None,
                     save_folder:Path=None, model=None, model_params:dict=None, 
                     remove_features:list[str]=None,
                     drop_engineer_source_features:bool=None):
    """
    Train a model with provided parameters or load saved final model.
    If a final model already exists, it is loaded. If not, a best grid search model is attempted to load.
    If neither exists, the model is trained from scratch using provided or default parameters.

    :param X: Training feature data. Optional if existing model is loaded.
    :param Y: Training target data. Optional if existing model is loaded.
    :param model_file_name: Name of the file to save/load the final model. If None, uses default from config.
    :param preprocessor_file_name: Name of the file preprocessor. If None, uses default from config.
    :param save_folder: Path to the folder where the model is saved or should be saved.
                        If None, defaults to the configured model directory.
    :param model: Optional custom model to train. If None, defaults to RegressorChain with XGBRegressor.
    :param model_params: Dictionary of model parameters for XGBRegressor. If None, uses best 
                        tuned parameters from config.
    :param remove_features: optional list of features to be removed during preprocessing. 
    If none, defaults to collinear feature list from config file.
    :param drop_engineer_source_features: handle source features after feature engineering.
                                         Default behavior-do not remove source features.
    :return: Tuple containing the trained model and the fitted preprocessor.
    """
    folder_path = save_folder or Path(__file__).parent.parent / cnfg["models"]["model_dir"]
    folder_path.mkdir(parents=True, exist_ok=True)
    model_file_name = model_file_name or cnfg["models"]["final_model_file"]
    preprocessor_file_name = preprocessor_file_name or cnfg["data"]["data_preprocessing"]["preprocessor_file"]
    hyper_tune_file = cnfg["hyperparameter_tuning"]["grid_search_file"]
    final_model_path = folder_path / model_file_name
    tuned_model_path = folder_path / hyper_tune_file
    preprocessor_path = folder_path / preprocessor_file_name
    preprocessor = _handle_preprocessor(X=X, preprocessor_path=preprocessor_path,remove_features=remove_features,
                                        drop_engineer_source_features=drop_engineer_source_features)

    if model_params and X is not None and Y is not None:
        chain_model = RegressorChain(XGBRegressor(**model_params))
        print("Fitting and saving final model from provided parameters.")  # optional, remove if not needed
        final_model = _handle_fitting(model=chain_model, X=X, Y=Y, preprocessor=preprocessor)
    elif model and X is not None and Y is not None:
        print("Fitting and saving provided custom model.")   # optional, remove if not needed
        final_model = _handle_fitting(model=model, X=X, Y=Y, preprocessor=preprocessor)
    elif final_model_path.is_file():
        final_model = joblib.load(filename=final_model_path) 
        print("Loading already saved final model.")  # optional, remove if not needed
    elif tuned_model_path.is_file():
        final_model = joblib.load(filename=tuned_model_path)
        print("Loading best grid model.")  # optional, remove if not needed
    elif cnfg["models"]["best_tuned_params"] and X is not None and Y is not None:
        chain_model = RegressorChain(XGBRegressor(**cnfg["models"]["best_tuned_params"]))
        print("Fitting and saving final model from config file parameters.")  # optional, remove if not needed
        final_model = _handle_fitting(model=chain_model, X=X, Y=Y, preprocessor=preprocessor)
    else:
        print("No parameters to fit final model or saved models found.")
        
    if not final_model_path.is_file():
        joblib.dump(value=final_model, filename=final_model_path)

    return final_model, preprocessor
