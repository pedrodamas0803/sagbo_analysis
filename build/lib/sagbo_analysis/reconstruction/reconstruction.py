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
        __init__ initializes the reconstruction object.

        Parameters
        ----------
        path : str
            path to the config file containing the information for the data processing for this sample/
        increment : int, optional
            chooses datasets to reconstruct at each increment-th dataset, by default 1
        sirt_iter : int, optional
            number of iterations for the SIRT algorithm; if 0, no SIRT volume will be calculated, by default 0
        PDHG_iter : int, optional
            number of iterations for the PDHG algorithm; if 0, no SIRT volume will be calculated, by default 0
        chunksize : int, optional
            number of slices to reconstruct at once in a chunked reconstruction, by default 512
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
        """
        shape_vwu returns the shape of the sinograms stack.

        Returns
        -------
        tuple
            shape of the sinograms stack following the numpy convention.
        """
        with h5py.File(self.processing_paths[0], "r") as hin:
            nz, ny, nx = hin["projections"].shape
        return ny, nz, nx

    @property
    def n_subvolumes(self):
        """
        n_subvolumes Calculates the numbre of subvolumes needed to reconstruct a full volume using an specified chunk size.

        Returns
        -------
        n : int
            number of subvolumes
        """
        n = self.shape_vwu[0] // self.chunksize
        r = self.shape_vwu[0] % self.chunksize
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
        """
        run_reconstruction runs the reconstruction using the selected algorithm for all the specified datasets in series. takes relatively long time.

        """

        for dataset in self.processing_paths:
            keys = self._get_h5_keys(path=dataset)
            if not self._is_valid_scan(h5_keys=keys):
                print("Your scan is not valid, skipping to the next.")
                continue

            print(
                f"Will reconstruct {get_dataset_name(dataset)} in {self.n_subvolumes} chunks."
            )

            data_vwu, angles, shifts, x0 = self._load_data(path=dataset)

            for ii in range(self.n_subvolumes):
                zmin, zmax = self._calc_chunk_index(index=ii)

                sub_data_vwu = self._divide_chunks(
                    data_vwu=data_vwu, zmin=zmin, zmax=zmax
                )

                if x0 == None or self.overwrite:
                    if "volFBP" in keys:
                        self._delete_entry(path=dataset, entry="volFBP")
                        print("Deleted FBP volume to reconstruct it.")
                    print(f"Reconstructing chunk {ii+1}.")
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
                data_vwu, angles, shifts, x0 = self._load_data(path=dataset)

                if self.overwrite and "volSIRT" in keys:
                    self._delete_entry(path=dataset, entry="volSIRT")
                    print("Deleted SIRT volume to reconstruct it.")

                for ii in range(self.n_subvolumes):
                    zmin, zmax = self._calc_chunk_index(index=ii)

                    sub_data_vwu = self._divide_chunks(
                        data_vwu=data_vwu, zmin=zmin, zmax=zmax
                    )

                    sub_x0 = self._divide_chunks_x0(x0=x0, zmin=zmin, zmax=zmax)

                    print(f"Reconstructing chunk {ii+1}.")

                    sub_SIRT = self._reconstruct_SIRT(
                        sinograms=sub_data_vwu,
                        angles_rad=angles,
                        shifts=shifts,
                        x0=sub_x0,
                    )

                    self._save_sub_volumes(index=ii, vol=sub_SIRT, path=dataset)

                    print(f"Reconstructed and wrote chunk {ii+1 } to file.")
            if self.PDHG_iter > 0:
                data_vwu, angles, shifts, x0 = self._load_data(path=dataset)

                if self.overwrite and "volPDHG" in keys:
                    self._delete_entry(path=dataset, entry="volPDHG")
                    print("Deleted PDHG volume to reconstruct it.")

                for ii in range(self.n_subvolumes):
                    zmin, zmax = self._calc_chunk_index(index=ii)

                    sub_data_vwu = self._divide_chunks(
                        data_vwu=data_vwu, zmin=zmin, zmax=zmax
                    )

                    sub_x0 = self._divide_chunks_x0(x0=x0, zmin=zmin, zmax=zmax)

                    print(f"Reconstructing chunk {ii+1}.")

                    sub_PDHG = self._reconstruct_PDHG(
                        sinograms=sub_data_vwu,
                        angles_rad=angles,
                        shifts=shifts,
                        x0=sub_x0,
                    )

                    self._save_sub_volumes(index=ii, vol=sub_PDHG, path=dataset)

                    print(f"Reconstructed and wrote chunk {ii+1} to file.")

            print("Going to the next volume ! ")

    def _reconstruct_FBP(
        self, sinograms: np.ndarray, angles_rad: np.array, shifts: np.ndarray
    ) -> np.ndarray:
        """
        _reconstruct_FBP reconstructs a tomography volume using a Filtered Back-Projection

        Parameters
        ----------
        sinograms : np.ndarray
            stack of sinograms to be used for the reconstruction
        angles_rad : np.array
            array of angles corresponding to each line of a sinogram
        shifts : np.ndarray
            array of detector shifts calculated using tomogrpahic consistency

        Returns
        -------
        volFBP : np.ndarray
            reconstructed volume
        """
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
    ) -> np.ndarray:
        """
        _reconstruct_SIRT reconstructs a volume using the SIRT algorithm

        Parameters
        ----------
        sinograms : np.ndarray
            stack of sinograms
        angles_rad : np.array
            array of angles referring to each line of one sinogram
        shifts : np.ndarray
            shifts array calculated from the tomographic consistency algorithm
        x0 : np.ndarray
            initialization volume for the reconstruction problem, usually the FBP volume.

        Returns
        -------
        volSIRT : np.ndarray
            reconstructed volume
        """
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
    ) -> np.ndarray:
        """
        _reconstruct_PDHG reconstructs a volume using the PDHG strategy

        Parameters
        ----------
        sinograms : np.ndarray
            stack of sinograms
        angles_rad : np.array
            array of angles referring to each line of one sinogram
        shifts : np.ndarray
            shifts array calculated from the tomographic consistency algorithm
        x0 : np.ndarray
            initialization volume for the reconstruction problem, usually the FBP volume.

        Returns
        -------
        volPDHG : np.ndarray
            reconstructed volume
        """
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
        """
        _load_data loads the data from the h5 file containing the corrected frames to be used in the reconstruction.

        Parameters
        ----------
        path : str
            absolute path to the h5 file from where the data will be retrieved.

        Returns
        -------
        data_vwu : np.ndarray
            stack of sinograms to be used in the reconstruction
        angles : np.array
            array of angular positions (in radians) related to each line of a sinogram in data_vwu.
        shifts : np.ndarray
            array containing the vertical and horizontal shifts calculated from the tomographic consistency algorithm
        x0 : None | np.ndarray
            None if there is no FBP volume; the FBP volume if exists.
        """
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
        """
        _get_h5_keys fetches the keys from an specified h5 file

        Parameters
        ----------
        path : str
            absolute path to the h5 file.

        Returns
        -------
        keys : list
            list containing the keys that exist in the h5 file.
        """
        with h5py.File(path, "r") as hin:
            keys = list(hin.keys())
        return keys

    def _is_return_scan(self, angles: np.array):
        """
        _is_return_scan determines if a scan was acquired in the reverse order (high to low angles) based on the rotation motor positions array.

        Parameters
        ----------
        angles : np.array
            array of angles fetched from the raw data h5 file containing the positions of the rotation motor.

        Returns
        -------
        bool
            returns True if the last angle is bigger than the first angle in the array.
        """
        if angles[0] > angles[-1]:
            return False
        else:
            return True

    def _calc_chunk_index(self, index: int):
        """
        _calc_chunk_index calculates the indexes for the chunked reconstruction of the index-th chunk.

        Parameters
        ----------
        index : int
            index of the index-th subvolume to be reconstructed.

        Returns
        -------
        (zmin, zmax) : tuple
            minimum and maximum indexes
        """
        zmin = index * self.chunksize
        zmax = (index + 1) * self.chunksize

        if zmax >= self.shape_vwu[0]:
            zmax = self.shape_vwu[0]

        return zmin, zmax

    def _divide_chunks(self, data_vwu: np.ndarray, zmin: int, zmax: int):
        """
        _divide_chunks divides the initial sinogram stack in chunks to be used in the chunked reconstruction.

        Parameters
        ----------
        data_vwu : np.ndarray
            entire stack of sinograms used to reconstruct a volume.
        zmin : int
            index of the first sinogram to be reconstructed within this chunk.
        zmax : int
            index of the last sinogram to be reconstructed within this chunk.

        Returns
        -------
        sub_data_vwu : np.ndarray
            chunk of sinograms to be used in the chunked recosntruction.
        """
        return data_vwu[zmin:zmax]

    def _save_sub_volumes(self, index: int, vol: np.ndarray, path: str):
        """
        _save_sub_volumes save the recosntructed subvolumes from the chunked reconstructed volume to the specified h5 file.

        Parameters
        ----------
        index : int
            index referrering to the index-th subvolume being reconstructed/saved.
        vol : np.ndarray
            index-th subvolume to be saved to file.
        path : str
            absolute path to the h5 file where the data should be saved.
        """
        with h5py.File(path, "a") as hout:
            hout[f"vol{index}"] = vol

    def _combine_subvolumes(self, path: str):
        """
        _combine_subvolumes combines the saved subvolumes from the chunked reconstruction in one final volume.

        Parameters
        ----------
        path : str
            absolute path to the h5 file where the subvolumes are stored.
        """
        with h5py.File(path, "a") as h:
            for ii in range(self.n_subvolumes):
                if ii == 0:
                    vol = h[f"vol{ii}"][:]
                else:
                    vol = np.concatenate((vol, h[f"vol{ii}"][:]))

            h["volFBP"] = vol

            for ii in range(self.n_subvolumes):
                del h[f"vol{ii}"]

    def _delete_entry(self, path: str, entry: str):
        """
        _delete_entry deletes the 'entry' from the h5 file located at 'path'.



        Parameters
        ----------
        path : str
            absolute path to the h5 file
        entry : str
            the h5 file entry to be deleted
        """
        with h5py.File(path, "a") as h:
            del h[entry]

    def _divide_chunks_x0(self, x0: np.ndarray, zmin: int, zmax: int):
        """
         Performs the division of the FBP volume to be used as initial gues for the iterative reconstruction methods.

         x0: np.ndarray; FBP volume loaded from the h5 file for the corresponding sample.
         zmin: int; lower index of the reconstructed slice to be used in the subvolume.
         zmax: int; maximum index of the reconstructed slice to be used in subvolume.

        Returns
        -------
        subvolume; np.ndarray; x0[zmin:zmax]
        """
        return x0[zmin:zmax]

    def _is_valid_scan(self, h5_keys: list):
        if (
            ("shifts" in h5_keys)
            and ("angles" in h5_keys)
            and ("projections" in h5_keys)
        ):
            return True
        else:
            return False
