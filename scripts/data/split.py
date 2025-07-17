from pathlib import Path
import pandas as pd
from scripts.config import cnfg
import json
from sklearn.model_selection import train_test_split

def _handle_timestamps(value):
    """
    A helper function to check for timestamps and convert them to strings.
    """
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def split_dataset(data:pd.DataFrame, validation_size:float=0.2,
                  n_test_samples:int=None,
                  test_set_folder = None,
                  test_set_filename = None,
                  target_columns = None)->tuple:
    """
    Split the input dataset into training and validation sets. Optionally sample a specified number
    of rows for a separate API functionality test and save this subset as a JSON file.
    Serialize timestamps if any in the API test set. 

    :param data: The input pandas DataFrame containing features and target columns.
    :param validation_size: Fraction of the remaining data to use as the validation set (default 0.2).
    :param n_test_samples: Number of random rows to sample for the test set.
    :param test_set_folder: Directory where the test set JSON file should be saved. Defaults to config file value.
    :param test_set_filename: Name of the test set JSON file. Defaults to filename from config file.
    :param target_columns: List of target column names. Defaults to config file target column list.
    :return: Tuple (X_train, X_valid, y_train, y_valid) with split features and targets.
    """
    target_columns = target_columns or cnfg["data"]["target_columns"]
    n_samples = n_test_samples
    path_to_folder = test_set_folder or Path(__file__).parent.parent / cnfg["data"]["data_dir"]
    filename = test_set_filename or cnfg["data"]["test_data_filename"]
    path_to_file = path_to_folder / filename
    if n_samples is not None:
        sample = data.sample(n=n_samples)
        data = data.drop(sample.index)
        feature_target_pairs = []
        for row in sample.iterrows():
            feature_target_pairs.append(
                {"features": {key: _handle_timestamps(value) for key, value in row[1].drop(target_columns).items()},
                "targets": row[1][target_columns].to_dict()}
            )
        with open(path_to_file, "w") as file:
            json.dump(obj=feature_target_pairs, fp=file, indent=2)


    X_train, X_valid, y_train, y_valid = train_test_split(
        data.drop(columns=target_columns),
        data[target_columns],
        test_size=validation_size,)
    
    return X_train, X_valid, y_train, y_valid
