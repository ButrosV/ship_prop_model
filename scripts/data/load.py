
import pandas as pd
from pathlib import Path
import gdown
import json

from scripts.config import cnfg


def get_and_load_file(
        data_folder:Path=None,
        filename:str=None,
        url:str=None, download:bool=True)->pd.DataFrame:
    """
    Load JSON dataset into a DataFrame. 
    Download JSON dataset from a Google Drive URL, if not already present.
    :param data_folder: Path to the folder where the file should be stored or loaded from.
    :param filename: Name of the file to check, download, and load.
    :param url: Google Drive URL for downloading the file if it's not found locally.
    :param download: Toogle off download if lile not present in the directory. 
                        Default behavior download.
    :return: A pandas DataFrame containing the data from the JSON file.
    """
    data_folder = data_folder or cnfg["data"]["data_dir"]
    file_name = filename or cnfg["data"]["data_filename"]
    url = url or cnfg["data"]["data_url"]
    data_folder.mkdir(parents=True, exist_ok=True)
    local_file_path = data_folder / file_name
    if not local_file_path.is_file():
        if download:
            print(f"Source data file not in {data_folder}, downloading file.")
            gdown.download(url, str(local_file_path), fuzzy=True, quiet=False)  # quiet=True after testing
        else:
            print(f"No {file_name} in {data_folder}. Returning None.")
            return None

    with open(local_file_path, 'r', encoding='utf-8') as f:  # more memory safe than pd.read_json(local_file_path)
        data = json.load(f)

    df = pd.DataFrame(data)

    return df
