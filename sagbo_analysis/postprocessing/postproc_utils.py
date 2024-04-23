import configparser
import os


def read_config_file(path: str):

    cfg = configparser.ConfigParser()
    cfg.read(path)

    cfg_dict = {
        'processing_dir': cfg.get('DIRECTORIES', 'processing_dir'),
        'datasets': [path for _, path in cfg.items('DATASETS')],

    }

    return cfg_dict


def get_dataset_name(path: str):

    return os.path.splitext(path)[0].split('/')[-1]

def build_tiff_path(path:str):

    return os.path.splitext(path)[0]+'.tiff'

def build_mask_path(path:str):

    return os.path.splitext(path)[0]+'_mask.tiff'