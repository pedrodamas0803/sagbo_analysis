import os

import h5py
import numpy as np

from .postproc_utils import get_dataset_name, read_config_file


class PostProcessing:

    def __init__(self, path: str, increment: int = 1):

        cfg = read_config_file(path)

        self.processing_dir = cfg['processing_dir']
        self.datasets = cfg['datasets']
        self.increment = increment

    @property
    def selected_datasets(self):
        datasets = []
        for ii, dataset in enumerate(self.datasets):
            if ii % self.increment == 0:
                datasets.append(dataset)
        return datasets

    @property
    def processing_paths(self):
        proc_paths = []
        for dataset in self.selected_datasets:

            name = get_dataset_name(dataset)
            path_to_process = os.path.join(
                self.processing_dir, name, f'{name}.h5')
            proc_paths.append(path_to_process)
        return proc_paths

    def _load_volume(self, path: str):

        with h5py.File(path, 'r') as hin:

            if 'volSIRT' in hin.keys():
                vol = hin['volSIRT'][:].astype(np.float32)
            else:
                vol = hin['volFBP'][:].astype(np.float32)
        return vol
