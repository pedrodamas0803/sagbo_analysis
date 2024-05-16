import os
import numpy as np

from .archive_utils import read_config_file

class MemorySaver:

    def __init__(self, path:str, increment:int =1):

        cfg = read_config_file(path=path)

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