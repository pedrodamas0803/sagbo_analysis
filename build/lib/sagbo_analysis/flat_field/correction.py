import os
import concurrent.futures

import h5py
import numpy as np
from PCA_flats.PCA_flats import PCAFlatFromFile

from .ff_utils import get_dataset_name, read_config_file, sagbo_mask


class FlatFieldCorrection:

    """
    Class that reads the PCA master file for the time-series and applies the corrections to selected datasets.
    """

    def __init__(self, path: str, increment: int = 1, mask: np.ndarray = None):
        """
        Inputs:

        path: str - path to the configuration file created by SampleInfo.
        increment: int - the increment for the processing of datasets within the time-series.

        """

        cfg = read_config_file(path)

        self.pca_flat_file = cfg["pca_flat_file"]
        self.flats_path = cfg["flats_path"]
        self.datasets = cfg["datasets"]
        self.projs_entry = cfg["projs_entry"]
        self.angles_entry = cfg["angles_entry"]
        self.processing_dir = cfg["processing_dir"]
        self.flats_entry = cfg["flats_entry"]
        self.increment = increment
        self.overwrite = False
        if cfg["overwrite"] == "True":
            self.overwrite = True
        if mask is not None:
            self.mask = mask

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
            path_to_save = os.path.join(self.processing_dir, name, f"{name}.h5")
            saving_paths.append(path_to_save)
        return saving_paths

    def run_correction(self, prop: float = 0.125, radius: float = None):
        """
        Method that runs the correction for the selected datasets.

        TODO: make it write the script to be launched by sbatch in several machines.
        """
        if self._check_components():
            pca = PCAFlatFromFile(path=self.pca_flat_file)
            if radius is not None:
                mask = sagbo_mask(radius, prop, pca.mean.shape)
        else:
            print(
                "No PCA decomposition found, defaulting to regular flat-field correction."
            )

        if self._check_components():
            for selected_dataset, saving_path in zip(
                self.selected_datasets, self.saving_paths
            ):
                assert selected_dataset != saving_path

                if not self._check_corrected_projections(saving_path):
                    print(f"Will correct {get_dataset_name(selected_dataset)}.")

                    projections, angles = self._load_proj_stack(selected_dataset)

                    if radius is not None:
                        pca.correct_stack(projections, save_path=saving_path, mask=mask)
                    else:
                        pca.correct_stack(
                            projections, save_path=saving_path, xprop=prop
                        )

                    with h5py.File(saving_path, "a") as hout:
                        hout["angles"] = angles
                elif self.overwrite:
                    print(
                        f"Corrected images were found for {get_dataset_name(saving_path)}, overwriting."
                    )
                    projections, angles = self._load_proj_stack(selected_dataset)

                    if radius is not None:
                        pca.correct_stack(projections, save_path=saving_path, mask=mask)
                    else:
                        pca.correct_stack(
                            projections, save_path=saving_path, xprop=prop
                        )

                    with h5py.File(saving_path, "a") as hout:
                        hout["angles"] = angles
                else:
                    continue
        else:
            for selected_dataset, saving_path in zip(
                self.selected_datasets, self.saving_paths
            ):
                if not self._check_corrected_projections(saving_path):
                    projections, angles = self._load_proj_stack(selected_dataset)
                    flat, dark = self._load_flats_darks()

                    projections -= dark
                    flat -= dark

                    corr_projs = -np.log(projections / flat)

                    with h5py.File(saving_path, "a") as hout:
                        hout["projections"] = corr_projs
                        hout["angles"] = angles
                    print(f"Corrected dataset and saved to {saving_path}.")

    def _load_proj_stack(self, path: str):
        with h5py.File(path, "r") as hin:
            projs = hin[self.projs_entry][:].astype(np.float32)
            angles = hin[self.angles_entry][:]

        return projs, angles

    def _check_corrected_projections(self, path: str):
        if os.path.exists(path):
            with h5py.File(path, "r") as hin:
                if "projections" in hin.keys() and "angles" in hin.keys():
                    if hin["projections"].shape[0] == hin["angles"].shape[0]:
                        return True
                else:
                    return False
        else:
            return False

    def _check_components(self):
        with h5py.File(self.pca_flat_file, "r") as hin:
            keys = list(hin.keys())

        if "p_components" in keys:
            return True
        else:
            return False

    def _load_flats_darks(self):
        with h5py.File(self.pca_flat_file, "r") as hin:
            flat = hin["mean"][:].astype(np.float32)
            dark = hin["dark"][:].astype(np.float32)

        return flat, dark

    # def _regular_flat_correction_par(self, projections: np.ndarray):
    #     def flat_correction(proj: np.ndarray, flat: np.ndarray, dark: np.ndarray):
    #         proj -= dark
    #         flat -= dark
    #         return -np.log(proj / flat)

    #     corr_projs = np.zeros_like(projections)

    #     with concurrent.futures.ProcessPoolExecutor() as pool:
    #         for ii, proj in pool.map(flat_correction, projections):
    #             corr_projs[ii] = proj

    #     return corr_projs
