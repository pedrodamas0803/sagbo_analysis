
import h5py
import numpy as np
from PCA_flats.PCA_flats import PCAFlatImages

from .ff_utils import read_config_file


class DecompositionPCA:

    ''' Class that runs the PCA decomposition from a collection of flat-fields and darks and saves it to a master PCA file.'''

    def __init__(self, config_path: str):
        '''
        Inputs:

        config_path: str - path to the configuration file defined by the class SampleInfo.

        '''

        cfg = read_config_file(config_path)

        self.darks_path = cfg['darks_path']
        self.flats_path = cfg['flats_path']
        self.pca_flat_file = cfg['pca_flat_file']
        # self.processing_dir = cfg['processing_dir']
        self.flats_entry = cfg['flats_entry']
        self.darks_entry = cfg['darks_entry']
        # self.datasets = cfg['datasets']

    @property
    def flats(self):
        flats = None
        try:
            with h5py.File(self.flats_path, 'r') as hin:
                flats = hin[self.flats_entry][:].astype(np.float32)
                print(f'Loaded {flats.shape[0]} flats.')
        except Exception as e:
            print(e)
            print(
                'Something went wrong while loading your flats. \n Check your flats entry.')
        return flats

    @property
    def darks(self):
        darks = None
        try:
            with h5py.File(self.darks_path, 'r') as hin:
                darks = hin[self.darks_entry][:].astype(np.float32)
                print(f'Loaded {darks.shape[0]} darks.')
        except Exception as e:
            print(e)
            print(f'Something went wrong while loading darks.\n Check your darks entry.')
        return darks

    def run_decomposition(self):
        ''' Runs the decomposition and saves it to the path given in the configuration file.'''

        pca = PCAFlatImages(self.flats, self.darks)
        pca.makeframes()
        pca.setdark()
        pca.setmask()
        pca.save_decomposition(path=self.pca_flat_file)
