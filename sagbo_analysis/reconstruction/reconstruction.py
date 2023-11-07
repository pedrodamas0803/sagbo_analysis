import os

# import concurrent.futures
# import time

import corrct as cct
import h5py
import numpy as np

from .reconstruction_utils import get_dataset_name, read_config_file


class Reconstruction:

    """Class that runs the reconstruction of the selected datasets, by default performs only FBP reconstruction."""

    def __init__(
        self,
        path: str,
        increment: int = 1,
        sirt_iter: int = 0,
        PDHG_iter: int = 0,
        chunksize: int = 512,
    ):
        """
        Inputs
        path: str - path to the configuration file written by SampleInfo.
        increment: int - te increment between datasets to be processed within the time-series data.
        sirt_iter: int = 0 - if it's set to a number bigger than 0 it will enable a SIRT reconstruction with the given number of iterations using the FBP volume as initial guess.

        """

        cfg = read_config_file(path)
        self.datasets = cfg["datasets"]
        self.processing_dir = cfg["processing_dir"]
        self.increment = increment
        self.sirt_iter = sirt_iter
        self.PDHG_iter = PDHG_iter
        self.overwrite = False
        if cfg["overwrite"] == "True":
            self.overwrite = True
        self.chunksize = chunksize

    @property
    def selected_datasets(self):
        datasets = []
        for ii, dataset in enumerate(self.datasets):
            if ii % self.increment == 0:
                datasets.append(dataset)
        return datasets

    @property
    def shape_vwu(self):
        with h5py.File(self.selected_datasets[0], "r") as hin:
            nz, ny, nx = hin["projections"].shape
        return ny, nz, nx

    @property
    def n_subvolumes(self):
        n = self.shape_vwu[1] // self.chunksize
        r = self.shape_vwu[1] % self.chunksize
        if r == 0:
            return n
        else:
            return n + 1

    @property
    def processing_paths(self):
        proc_paths = []
        for dataset in self.selected_datasets:
            name = get_dataset_name(dataset)
            path_to_process = os.path.join(self.processing_dir, name, f"{name}.h5")
            proc_paths.append(path_to_process)
        return proc_paths

    def run_reconstruction(self):
        """Method that runs the reconstructions with the given parameters."""

        for dataset in self.processing_paths:
            print(f"Will reconstruct {get_dataset_name(dataset)}.")

            data_vwu, angles, shifts, x0 = self._load_data(path=dataset)

            for ii in range(self.n_subvolumes):
                zmin, zmax = self._calc_chunk_index(index=ii)

                sub_data_vwu = self._divide_chunks(
                    data_vwu=data_vwu, zmin=zmin, zmax=zmax
                )

                keys = self._get_h5_keys(path=dataset)

                if x0 == None or self.overwrite:
                    print(f"Will reconstruct FBP volume in {self.n_subvols} chunks.")
                    subFBP = self._reconstruct_FBP(
                        sinograms=sub_data_vwu, angles_rad=angles, shifts=shifts
                    )
                    self._save_sub_volumes(index=ii, vol=subFBP, path=dataset)

                    print(f"Reconstructed and wrote chunk {ii+1} to file.")

                else:
                    break

            self._combine_subvolumes(path=dataset)

            print(f"Recombined all subvolumes and saved to file.")

            if self.sirt_iter > 0:
                if "volSIRT" in keys and not self.overwrite:
                    print(
                        "SIRT volume found, skipping to the next. Set overwrite flag to True if you want to reconstruct it again."
                    )
                    continue

                solverSIRT = cct.solvers.Sirt(verbose=False)
                with cct.projectors.ProjectorUncorrected(
                    vol_geom, angles_rad, prj_geom=proj_geom
                ) as A:
                    volSIRT, _ = solverSIRT(
                        A, data_vwu, x0=volFBP, iterations=self.sirt_iter
                    )

                with h5py.File(dataset, "a") as hout:
                    if "volSIRT" in hout.keys():
                        del hout["volSIRT"]
                    hout["volSIRT"] = volSIRT
                print("Reconstructed SIRT volume and wrote it to file.")

                del volSIRT, solverSIRT

            if self.PDHG_iter > 0:
                if "volPDHG" in keys and not self.overwrite:
                    print(
                        "PDHG volume found, skipping to the next. Set overwrite flag to True if you want to reconstruct it again."
                    )
                    continue

                solverPDHG = cct.solvers.PDHG(verbose=True)
                with cct.projectors.ProjectorUncorrected(
                    vol_geom, angles_rad, prj_geom=proj_geom
                ) as A:
                    volPDHG, _ = solverPDHG(
                        A, data_vwu, x0=volFBP, iterations=self.PDHG_iter
                    )

                with h5py.File(dataset, "a") as hout:
                    if "volPDHG" in hout.keys():
                        del hout["volPDHG"]
                    hout["volPDHG"] = volPDHG
                print("Reconstructed PDHG volume and wrote it to file.")

                del volPDHG, solverPDHG

            del data_vwu, angles_rad, shifts, volFBP, vol_geom, proj_geom
            print("Going to the next volume ! ")

    def _reconstruct_FBP(
        self, sinograms: np.ndarray, angles_rad: np.array, shifts: np.ndarray
    ):
        ang0 = angles_rad[0]
        angles_rad = angles_rad - ang0

        solverFBP = cct.solvers.FBP(verbose=False, fbp_filter="hann")
        vol_geom = cct.models.VolumeGeometry.get_default_from_data(sinograms)
        proj_geom = cct.models.ProjectionGeometry.get_default_parallel()
        proj_geom.set_detector_shifts_vu(shifts)

        with cct.projectors.ProjectorUncorrected(
            vol_geom, angles_rot_rad=angles_rad, prj_geom=proj_geom
        ) as A:
            volFBP, _ = solverFBP(A, sinograms, iterations=10)

        return volFBP

    def _reconstruct_SIRT(
        self,
        sinograms: np.ndarray,
        angles_rad: np.array,
        shifts: np.ndarray,
        x0: np.ndarray,
    ):
        ang0 = angles_rad[0]
        angles_rad = angles_rad - ang0

        solverSIRT = cct.solvers.Sirt(verbose=False)
        vol_geom = cct.models.VolumeGeometry.get_default_from_data(sinograms)
        proj_geom = cct.models.ProjectionGeometry.get_default_parallel()
        proj_geom.set_detector_shifts_vu(shifts)

        with cct.projectors.ProjectorUncorrected(
            vol_geom, angles_rot_rad=angles_rad, prj_geom=proj_geom
        ) as A:
            volSIRT, _ = solverSIRT(A, sinograms, x0=x0, iterations=self.sirt_iter)

        return volSIRT

    def _reconstruct_PDHG(
        self,
        sinograms: np.ndarray,
        angles_rad: np.array,
        shifts: np.ndarray,
        x0: np.ndarray,
    ):
        ang0 = angles_rad[0]
        angles_rad = angles_rad - ang0

        solverPDHG = cct.solvers.PDHG(verbose=False)
        vol_geom = cct.models.VolumeGeometry.get_default_from_data(sinograms)
        proj_geom = cct.models.ProjectionGeometry.get_default_parallel()
        proj_geom.set_detector_shifts_vu(shifts)

        with cct.projectors.ProjectorUncorrected(
            vol_geom, angles_rot_rad=angles_rad, prj_geom=proj_geom
        ) as A:
            volPDHG, _ = solverPDHG(A, sinograms, x0=x0, iterations=self.PDHG_iter)

        return volPDHG

    def _load_data(self, path: str):
        with h5py.File(path, "a") as hin:  # dangerous
            x0 = None
            if self.PDHG_iter > 0 or self.sirt_iter > 0:
                try:
                    x0 = hin["volFBP"][:]
                except Exception:
                    print("FBP volume not found. Reconstruct it first.")

            angles = hin["angles"][:]
            projs = hin["projections"][:].astype(np.float32)
            shifts = hin["shifts"][:]

            if self._is_return_scan(angles=angles):
                projs = np.flip(projs, axis=0)
                angles = np.flip(angles, axis=0)
                shifts = np.flip(shifts, axis=0)

        return np.rollaxis(projs, 1, 0), np.deg2rad(angles), shifts, x0

    def _get_h5_keys(self, path: str):
        with h5py.File(path, "r") as hin:
            keys = list(hin.keys())
        return keys

    def _is_return_scan(self, angles: np.ndarray):
        if angles[0] > angles[-1]:
            return True
        else:
            return False

    def _calc_chunk_index(self, index: int):
        zmin = index * self.chunksize
        zmax = (index + 1) * self.chunksize

        if zmax >= self.shape_vwu[1]:
            zmax = self.shape_vwu[1]

        return zmin, zmax

    def _divide_chunks(self, data_vwu: np.ndarray, zmin: int, zmax: int):
        return data_vwu[zmin:zmax]

    def _save_sub_volumes(self, index: int, vol: np.ndarray, path: str):
        with h5py.File(path, "a") as hout:
            hout[f"vol{index}"] = vol

    def _combine_subvolumes(self, path: str):
        vol = np.array([])
        with h5py.File(path, "a") as h:
            for ii in range(self.n_subvolumes):
                vol = np.concatenate(h[f"vol{ii}"][:], axis=0)

            h["volFBP"] = vol

            for ii in range(self.n_subvolumes):
                del vol[f"vol{ii}"]
