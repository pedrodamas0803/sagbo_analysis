
import numpy as np
import h5py
from PCA_flats.PCA_flats import PCAFlatImages

from .utilsff import read_config_file


class FlatFieldCorrection:

    def __init__(self, config_path:str):

        cfg = read_config_file(config_path)

        self.darks_path = cfg['darks_path']
        self.pca_flat_file = cfg['pca_flat_file']
        self.processing_dir = cfg['processing_dir']
        self.flats_entry = cfg['flats_entry']
        self.darks_entry = cfg['darks_entry']
        self.datasets = cfg['datasets']

    @property
    def flats(self):
        for ii, dataset in enumerate(self.datasets):
            try:
                if ii == 0:
                    with h5py.File(dataset, 'r') as hin:
                        flats = hin[self.flats_entry][:].astype(np.float32)
                else:
                    with h5py.File(dataset, 'r') as hin:
                        flats = np.concatenate([flats, hin[self.flats_entry][:].astype(np.float32)])
            except Exception as e:
                print(f'Dataset {dataset} could not be loaded, skipping to the next. \n {e}')
        return flats

    @property
    def darks(self):
        try:
            with h5py.File(self.darks_path, 'r') as hin:
                darks = hin[self.darks_entry][:].astype(np.float32)
        except Exception as e:
            print(f'Something went wrong while loading darks.\n {e}')
        return darks

    

    def run_decomposition(self):
        
        pca = PCAFlatImages(self.flats, self.darks)
        pca.make_frames()
        pca.set_dark()
        pca.set_masl()
        pca.save_decomposition(self.pca_flat_file)



