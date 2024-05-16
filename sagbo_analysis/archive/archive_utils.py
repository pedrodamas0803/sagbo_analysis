import os
import configparser


def read_config_file(path: str):
    cfg = configparser.ConfigParser()
    cfg.read(path)

    cfg_dict = {
        'processing_dir': cfg.get('DIRECTORIES', 'processing_dir'),
        'datasets': [path for _, path in cfg.items('DATASETS')],
        'load_entry': cfg.get('ENTRIES', 'load'),
        'proj_entry': cfg.get('ENTRIES', 'projections')

    }

    return cfg_dict


def get_dataset_name(path: str):
    return os.path.splitext(path)[0].split('/')[-1]
