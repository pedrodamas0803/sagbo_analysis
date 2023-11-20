import configparser
import os


def read_config_file(path: str):
    """
    read_config_file gets relevant information from the config file.

    Parameters
    ----------
    path : str
        path to the config file.

    Returns
    -------
    dict
        dictionary containg the information for the recnstruction step.
    """

    cfg = configparser.ConfigParser()
    cfg.read(path)

    cfg_dict = {
        "processing_dir": cfg.get("DIRECTORIES", "processing_dir"),
        "datasets": [path for _, path in cfg.items("DATASETS")],
        # 'dering' : cfg.get('FLAGS', 'dering'),
        "overwrite": cfg.get("FLAGS", "overwrite"),
    }

    return cfg_dict


def get_dataset_name(path: str):
    return os.path.splitext(path)[0].split("/")[-1]
