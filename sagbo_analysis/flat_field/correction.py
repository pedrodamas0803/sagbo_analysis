import os

import h5py
import numpy as np
from PCA_flats.PCA_flats import PCAFlatFromFile

from .ff_utils import get_dataset_name, read_config_file


class FlatFieldCorrection:

    def __init__(self, path: str, increment=1):

        cfg = read_config_file(path)

        self.pca_flat_file = cfg['pca_flat_file']
        self.datasets = cfg['datasets']
        self.projs_entry = cfg['projs_entry']
        self.angles_entry = cfg['angles_entry']
        self.processing_dir = cfg['processing_dir']
        self.increment = increment

    @property
    def selected_datasets(self):
        datasets = []
        for ii, dataset in enumerate(self.datasets):
            if ii % self.increment == 0:
                datasets.append(dataset)
        return datasets

    # TODO
    def run_correction(self):

        pca = PCAFlatFromFile(path=self.pca_flat_file)

        for dataset in self.selected_datasets:

            name = get_dataset_name(dataset)
            path_to_save = os.path.join(
                self.processing_dir, name, f'{name}.h5')
            print(name)
            print(path_to_save)
            # try:
            #     projs, angles = self._load_proj_stack(dataset)

            #     assert path_to_save != dataset

            #     if os.path.exists(path_to_save) and path_to_save != dataset:
            #         print(f'Will remove {path_to_save}')
            #         old_dir = os.getcwd()
            #         os.chdir(os.path.dirname(path_to_save))
            #         os.remove(f'{name}.h5')
            #         os.chdir(old_dir)

            #     pca.correct_stack(projections=projs, save_path=path_to_save)

            #     with h5py.File(path_to_save, 'a') as hout:

            #         hout['angles'] = angles

            # except Exception as e:
            #     print(e)
            #     print(
            #         f'Something went wrong with {dataset} correction.\n Nothing was saved to {path_to_save}')

    def _load_proj_stack(self, path: str):

        with h5py.File(path, 'r') as hin:

            projs = hin[(self.projs_entry)][:].astype(np.float32)
            angles = hin[self.angles_entry][:]

        return projs, angles
