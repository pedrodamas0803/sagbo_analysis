
# from dataclasses import dataclass
import configparser
import os
import h5py


def generate_config_file():

    config = configparser.ConfigParser()

    config['DIRECTORIES'] = {
        'Acquisition_directory' : '',
        'Folder_basename' : '',
        'PCA_file': ''
    }

    # config['PCA_DECOMPOSITION'] = {
    #     ''
    # }

    config['RECONSTRUCTION'] = {
        'Number_of_acquisitions' : 1,
        'Slab_size' : 256,
        'Do_SIRT' : False
    }

    config['ENTRIES'] = {
        'Projections' : '',
        'Flats' : '',
        'Darks' : ''

    }

    return config

def write_config_file(config: configparser.ConfigParser, path: str = '/'):

    if isinstance(config, configparser.ConfigParser):

        with open('config.ini', 'w') as configfile:

            config.write(configfile)

def read_config_file(path:str):

    if path.endswith('ini'):

        config = configparser.ConfigParser()

        config.read(path)
    
    return config





    


if __name__ == '__main__':

    cf = generate_config_file()

    write_config_file(cf, path = 'tests/')

    del cf

    cf = read_config_file('config.ini')

    print(cf['RECONSTRUCTION']['slab_size'])

    os.remove('config.ini')