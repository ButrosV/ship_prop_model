import argparse
import requests # pyright: ignore[reportMissingModuleSource]
import json
from random import randint

from scripts.config import cnfg

LIMITED_FEATURE_FILE = cnfg["data"]["data_dir"] / cnfg["data"]["test_data_filename_short"]
FULL_FEATURE_FILE = cnfg["data"]["data_dir"] / cnfg["data"]["test_data_filename"]
URL = f"http://{cnfg["api"]["host"] }:{str(cnfg["api"]["port"])}{cnfg["api"]["predict_endpoint"]}"


def main(all_features:bool=False, url=URL, source_file=None):
    """
    Send a randomized test sample to a FastAPI prediction endpoint and compare response to ground truth.

    Load a single data sample (features and targets) from either a limited or full feature test set 
    defined in the config. Sends the selected sample to the prediction API endpoint via POST request 
    and prints the API response, true values, and the index of the selected test sample.

    :param all_features: Flag indicating whether to use the full feature set. 
        If False, uses the limited feature file defined in config. Default is False.
    :param url: Full API URL to send the POST request to. 
        Defaults to value constructed from config.
    :param source_file: Optional override path to a JSON file with test data. 
        If None, path is selected based on `all_features` flag. 
        File has to comply with format having top level 'features' and 'targets' keys.
    :return: None. Prints the API response and true values for comparison.
    """
    if source_file is not None:
        path_to_json = source_file
    elif not all_features:
        path_to_json = LIMITED_FEATURE_FILE
    else:
        path_to_json = FULL_FEATURE_FILE

    with open(path_to_json, 'r') as file:  
        input_data = json.load(file)
    random_sample = randint(0, len(input_data) - 1)
    input_feats = input_data[random_sample]["features"]
    true_values= input_data[random_sample]["targets"]

    response = requests.post(url, json=input_feats)

    print(f"API response Status code: {response.status_code}")
    print(f"Predicted values (API call):\t{response.json()}")
    print(f"True predictions:\t\t{true_values}")
    print(f"Randomized test sample index for true values: {random_sample}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_features", action="store_true",
                        help="Test predictions with full feature set.")
    parser.add_argument("--source_file", type=str, default=None,
                        help="Optional path to a custom JSON input file.")
    
    args = parser.parse_args()
    main(all_features=args.all_features, source_file=args.source_file)
    