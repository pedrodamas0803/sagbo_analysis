import configparser

def read_config_file(path:str):

    cfg = configparser.ConfigParser()
    cfg.read(path)

    cfg_dict = {
        'darks_path': cfg.get('DIRECTORIES', 'darks_path'),
        'pca_flat_file':cfg.get('DIRECTORIES', 'pca_flat_file'),
        'processing_dir':cfg.get('DIRECTORIES', 'processing_dir'),
        'flats_entry':cfg.get('ENTRIES', 'flats'), 
        'darks_entry':cfg.get('ENTRIES','darks'),
        'datasets': cfg.get('DATASETS', 'datasets')
    }

    return cfg_dict
    

def _validate_config_file():
    pass


