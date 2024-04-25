import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage as sk

# import tifffile
from .dvc_utils import read_config_file, get_dataset_name
from .mscript import uncertainty_mesh_size, uncertainty_lambda_size


class DVC_Setup:

    """
    Class that reads a configuration file and sets up the directory structure to run DVC.

    """

    def __init__(self, config_file: str, increment: int = 1) -> None:
        cfg = read_config_file(path=config_file)

        self.processing_dir = cfg["processing_dir"]
        self.datasets = cfg["datasets"]
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
            path_to_process = os.path.join(self.processing_dir, name, f"{name}.h5")
            proc_paths.append(path_to_process)
        return proc_paths

    @property
    def dvc_dir(self):
        return os.path.join(self.processing_dir, "DVC_Analysis")

    @property
    def meshing_dir(self):
        return os.path.join(self.processing_dir, "meshing")

    @property
    def uncertainty_dir(self):
        return os.path.join(self.dvc_dir, "uncertainty")


    def build_folder_structure(self):
        try:
            os.mkdir(self.dvc_dir)
        except FileExistsError:
            print("DVC directory already exists, skipping.")
        try:
            os.mkdir(self.uncertainty_dir)
        except FileExistsError:
            print("Uncertainty folder already exists, skipping.")

        self._link_vtks()
        self._link_images()
        self._link_mask()

    def _link_mask(self):
        mask_path = glob.glob(os.path.join(self.meshing_dir, '*mask*'))

        for mask in mask_path:
            dst = os.path.join(self.dvc_dir, os.path.basename(mask))
            if not os.path.exists(dst):
                os.symlink(src=mask, dst=dst)
            else:
                os.remove(dst)
                os.symlink(src=mask, dst=dst)

    def _link_vtks(self):
        vtks = glob.glob(os.path.join(self.meshing_dir, "*.vtk"))

        for vtk in vtks:
            dst = os.path.join(self.dvc_dir, os.path.basename(vtk))
            if not os.path.exists(dst):
                os.symlink(src=vtk, dst=dst)
            else:
                os.remove(dst)
                os.symlink(src=vtk, dst=dst)

    def _link_images(self):
        for dataset in self.processing_paths:
            filename, _ = os.path.splitext(dataset)
            tiff_name = f"{filename}.tiff"
            dst = os.path.join(self.dvc_dir, os.path.basename(tiff_name))

            if not os.path.exists(dst):
                os.symlink(src=tiff_name, dst=dst)
            else:
                os.remove(dst)
                os.symlink(src=tiff_name, dst=dst)

