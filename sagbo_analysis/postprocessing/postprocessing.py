import os

import h5py
import numpy as np
from dvc_preprocessing.preprocessing import crop_around_CoM, volume_CoM
from skimage.exposure import rescale_intensity
from skimage.io import imsave

from ..utils import calc_color_lims
from .postproc_utils import build_tiff_path, get_dataset_name, read_config_file


class PostProcessing:

    def __init__(self, path: str, increment: int = 1, prop=0.25, mult=1):

        cfg = read_config_file(path)

        self.processing_dir = cfg['processing_dir']
        self.datasets = cfg['datasets']
        self.increment = increment
        self.prop = prop
        self.mult = mult

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

    def run_postprocessing(self):

        for dataset in self.processing_paths:

            vol = self._load_volume(path=dataset)

            center_of_mass = volume_CoM(vol)

            cropped_vol = crop_around_CoM(
                vol, center_of_mass, xprop=self.prop, yprop=self.prop)

            imin, imax = calc_color_lims(cropped_vol, mult=self.mult)

            rescaled_img = rescale_intensity(
                cropped_vol, in_range=(imin, imax), out_range='uint8')

            save_path = build_tiff_path(dataset)

            imsave(save_path, rescaled_img,
                   plugin='tifffile', check_contrast=False)

    def _load_volume(self, path: str):

        with h5py.File(path, 'r') as hin:

            if 'volSIRT' in hin.keys():
                vol = hin['volSIRT'][:].astype(np.float32)
            else:
                vol = hin['volFBP'][:].astype(np.float32)
        return vol
