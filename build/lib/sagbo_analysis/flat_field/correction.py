import os

import h5py
import numpy as np
from PCA_flats.PCA_flats import PCAFlatFromFile

from .ff_utils import get_dataset_name, read_config_file


class FlatFieldCorrection:

    def __init__(self, path: str, increment=1):

        cfg = read_config_file(path)

        self.pca_flat_file = cfg['pca_flat_file']
        self.flats_path = cfg['flats_path']
        self.datasets = cfg['datasets']
        self.projs_entry = cfg['projs_entry']
        self.angles_entry = cfg['angles_entry']
        self.processing_dir = cfg['processing_dir']
        self.flats_entry = cfg['flats_entry']
        self.increment = increment

    @property
    def selected_datasets(self):
        datasets = []
        for ii, dataset in enumerate(self.datasets):
            if ii % self.increment == 0:
                datasets.append(dataset)
        return datasets

    @property
    def saving_paths(self):
        saving_paths = []
        for dataset in self.selected_datasets:

            name = get_dataset_name(dataset)
            path_to_save = os.path.join(
                self.processing_dir, name, f'{name}.h5')
            saving_paths.append(path_to_save)
        return saving_paths

    # TODO
    def run_correction(self):

        pca = PCAFlatFromFile(path=self.pca_flat_file)

        for selected_dataset, saving_path in zip(self.selected_datasets, self.saving_paths):

            assert selected_dataset != saving_path

            if not self._check_corrected_projections(saving_path):

                print(f'Will correct {get_dataset_name(selected_dataset)}.')

                projections, angles = self._load_proj_stack(selected_dataset)

                pca.correct_stack(projections, save_path=saving_path)

                with h5py.File(saving_path, 'a') as hout:
                    hout['angles'] = angles
            else:

                print(
                    f'Corrected images were found for {get_dataset_name(saving_path)}, skipping.')

    def _load_proj_stack(self, path: str):

        with h5py.File(path, 'r') as hin:

            if hin[self.flats_entry].shape[0] > 600:
                # self.projs_entry = self.flats_entry
                projs = hin[self.flats_entry][:].astype(np.float32)
                angles = np.arange(0, 360, 360/projs.shape[0])
            else:
                projs = hin[self.projs_entry][:].astype(np.float32)
                angles = hin[self.angles_entry][:]

        return projs, angles

    def _check_corrected_projections(self, path: str):

        if os.path.exists(path):

            with h5py.File(path, 'r') as hin:

                if 'projections' in hin.keys() and 'angles' in hin.keys():
                    if hin['projections'].shape[0] == hin['angles'].shape[0]:
                        return True
                else:
                    return False
        else:
            return False
