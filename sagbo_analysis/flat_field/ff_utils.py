
import configparser
import os

import numpy as np


def read_config_file(path: str):

    cfg = configparser.ConfigParser()
    cfg.read(path)

    cfg_dict = {
        'flats_path': cfg.get('DIRECTORIES', 'flats_path'),
        'darks_path': cfg.get('DIRECTORIES', 'darks_path'),
        'pca_flat_file': cfg.get('DIRECTORIES', 'pca_flat_file'),
        'processing_dir': cfg.get('DIRECTORIES', 'processing_dir'),
        'flats_entry': cfg.get('ENTRIES', 'flats'),
        'darks_entry': cfg.get('ENTRIES', 'darks'),
        'projs_entry': cfg.get('ENTRIES', 'projections'),
        'angles_entry': cfg.get('ENTRIES', 'angles'),
        'overwrite': cfg.get('FLAGS', 'overwrite'),
        'datasets': [path for _, path in cfg.items('DATASETS')]

    }

    return cfg_dict


def get_dataset_name(path: str):

    return os.path.splitext(path)[0].split('/')[-1]

def circular_mask(radius:float, shape:tuple):
    ny = np.arange(0, shape[0], 1)
    nx = np.arange(0, shape[1], 1)
    x, y = np.meshgrid(nx, ny)

    y -= (shape[0] // 2)
    x -= (shape[1] // 2)

    c = np.sqrt(x**2 + y**2)

    mask = c < radius

    return mask

def sample_mask(xprop:float, shape:tuple):

    imin = (shape[1] // 2) - int(xprop * shape[1])
    imax = (shape[1] // 2) + int(xprop * shape[1])
    mask = np.ones(shape, dtype=bool)

    mask[:, imin:imax] = False

    return mask

def sagbo_mask(radius, xprop, shape):

    """ 
    Gets a mask to remove the areas from the furnace and the area for the sample together.
    """

    return circular_mask(radius, shape) * sample_mask(xprop, shape)
    
