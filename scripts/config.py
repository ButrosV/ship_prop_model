from pathlib import Path
import yaml


def load_configuration(path:str=None)->dict:
    """
    Load YAML configuration file and convert directory paths to absolute Path objects relative to 
    current working directory. Thus call this function from one level bellow project root only.
    :param path: Relative or absolute path to the YAML configuration file.
    :return: A dictionary representing the configuration, with directory paths converted to Path objects.
    """
    if path is None:
        try:
            path = Path(__file__).resolve().parents[1] / "config.yaml"
        except NameError:
            path = Path().resolve().parents[1] / "config.yaml"
    else:
        path = Path(path)
        
    with open(path) as f:
        configuration = yaml.safe_load(f)
    for section in configuration.values():
        for key, val in section.items():
            if "dir" in key or "path" in key:
                try:
                    section[key] = Path(__file__).resolve().parents[1] / val
                except NameError:
                    section[key] = Path().resolve().parents[1] / val
    return configuration


cnfg = load_configuration()
