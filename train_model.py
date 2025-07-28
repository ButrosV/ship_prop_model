import argparse

from scripts.config import cnfg
from scripts.data.load import get_and_load_file
from scripts.data.clean import clean_data
from scripts.data.split import split_dataset
from scripts.model.evaluation import evaluation_tabel
from scripts.model.tuning import HyperParamSearch
from scripts.model.get_model import load_train_model

from scripts.utils.eda import top_correlations

# python train_model.py --tune_model or/and --task_features from CLI ta adjust flag/s.
def main(tune_model:bool=False, task_features:bool=False):
    """
    Run the full model pipeline: load data, split, preprocess, optionally tune model, train, and evaluate.

    :param tune_model: If True, runs full randomized + grid hyperparameter search
                       before training. If False, skips tuning and uses default
                       or previously saved model parameters.
    param task_features: If True, remove most features from data set according to 
                            original task description.
    :return: None
    """
    remove_features = cnfg["data"]["data_preprocessing"]["features_to_drop_task"] if task_features else None
    drop_engineer_source_features = False if task_features else True
    # Load data
    df = get_and_load_file()  # add data_folder=PATH_TO_DATA when testing

    # Clean data
    df = clean_data(data=df)

    # Split data
    X_train, X_test, y_train, y_test = split_dataset(data=df, n_test_samples=3)

    # (optional, uncomment if needed. Run time ~ 15h)
    if tune_model:
        tjuuneris = HyperParamSearch(remove_features=remove_features,
                                         drop_engineer_source_features=drop_engineer_source_features)
        tjuuneris.full_param_search(X=X_train, y=y_train, n_jobs=2)

    # Train or load model 
        the_model, the_preprocessor = tjuuneris.grid_search_result, tjuuneris.preprocess_output
    else:
        the_model, the_preprocessor = load_train_model(X=X_train, Y=y_train,
                                                       remove_features=remove_features,
                                                       drop_engineer_source_features=drop_engineer_source_features)
    # Evaluate
    y_pred = the_model.predict(the_preprocessor.transform(X_test))
    print(f"\nEvaluation table for XGBoost model with random->Grid hyperparameter search.")
    print(evaluation_tabel(
        predictions=y_pred,
        y_true=y_test,
        original_dataset_correlations=top_correlations(data=df)
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune_model", action="store_true",
                        help="Enable model tuning with random and grid searches if flag is passed")
    parser.add_argument("--task_features", action="store_true",
                        help="Use only shortlist of features from task description for model training")
    args = parser.parse_args()

    main(tune_model=args.tune_model, task_features=args.task_features)
