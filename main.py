from pathlib import Path  # remove after testing
import gc # remove after testing
import pandas as pd   # remove after testing
import argparse

from scripts.config import cnfg
from scripts.data.load import get_and_load_file
from scripts.data.clean import clean_data
from scripts.data.feature_select import FeatureSelector
from scripts.data.feature_engineer import FeatureEngineer
from scripts.model.evaluation import evaluation_tabel
from scripts.model.tuning import HyperParamSearch
from scripts.model.get_model import load_train_model

from sklearn.model_selection import train_test_split

# from scripts.utils.visual import compute_correlations_matrix
from scripts.utils.eda import top_correlations

# PATH_TO_MODEL = Path(r"C:\\Users\\Peteris\\Documents\\CV\\2025\\shiprojects\\hw_app\\models\\")  # remove after testing
# PATH_TO_MODEL = Path(r"C:\\Users\\pich\\Documents\\CV\\2025\\shiprojects\\hw_app\\models\\")  # remove after testing

PATH_TO_DATA = Path(r"C:\\Users\\Peteris\\Documents\\CV\\2025\\shiprojects\\hw_app\\data\\")  # remove after testing
# PATH_TO_DATA = Path(r"C:\\Users\\pich\\Documents\\CV\\2025\\shiprojects\\hw_app\\data\\")  # remove after testing


# python main.py --tune_model from CLI ta adjust flag
def main(tune_model:bool=False):
    """
    Run the full model pipeline: load data, preprocess, optionally tune model, train, and evaluate.

    :param tune_model: If True, runs full randomized + grid hyperparameter search
                       before training. If False, skips tuning and uses default
                       or previously saved model parameters.
    :return: None
    """
    # Load data
    df = get_and_load_file(data_folder=PATH_TO_DATA)  # remove PATH_TO_DATA after testing

    # Clean data
    df = clean_data(data=df)

    # Split data
    targets = cnfg["data"]["target_columns"]
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=targets),
        df[targets],
        test_size=0.2,
        random_state=42  # remove after testing
        )

    # Preprocess and search for best parameters for XGBoost model.
    # (optional, uncomment if needed. Run time ~ 15h)
    if tune_model:
        # tjuuneris = HyperParamSearch(save_folder=PATH_TO_MODEL)  # remove PATH_TO_MODEL after testing
        tjuuneris = HyperParamSearch()
        tjuuneris.full_param_search(X=X_train, y=y_train, n_jobs=2)

    # Train or load model 
        the_model, the_preprocessor = tjuuneris.grid_search_result, tjuuneris.preprocess_output
    else:
        the_model, the_preprocessor = load_train_model(X=X_train, y=y_train,
                                                    #    model_params=cnfg["models"]["best_tuned_params"],  # nicely working params, remove after testing
                                                    #    save_folder=PATH_TO_MODEL # remove PATH_TO_MODEL after testing
                                                       )
    # Evaluate
    y_pred = the_model.predict(the_preprocessor.transform(X_test))
    print(f"\nEvaluation table for XGBoost model with random->Grid hyperparameter search.")
    print(evaluation_tabel(
        predictions=y_pred,
        y_true=y_test,
        original_dataset_correlations=top_correlations(data=df)
        ))

    gc.collect()  # clean up unreachable objects in memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_model", action="store_true", help="Enable model tuning with random and grid searches if flag is passed")
    args = parser.parse_args()

    main(tune_model=args.tune_model)
