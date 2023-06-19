import configparser

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
