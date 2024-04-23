import os
import concurrent.futures
import time

from algotom.prep.removal import remove_all_stripe
import corrct as cct
import h5py
import numpy as np

from .align_utils import binning, get_dataset_name, read_config_file


class ProjectionAlignment:
    """Class to perform projection alignment based on tomographic consistency."""

    def __init__(self, path: str, increment: int = 1, iterations=5, slab_size=400):
        """
        Inputs:

        path: str -  the path to the configuration file where the data processing information is stored.
        increment: int - the step between each dataset to be processed among the time-series.
        iterations: int - number of iterations for the tomographic consistency alignment. Higher tends to be better, but takes longer.
        slab_size: int - number of slices, centered in the height of the projections, to be used for the alignment. Higher tends to be better, the data will be binned for the calculationsfor the calculations, don't choose a number too small.

        The default values tend to work well, unless the error-motion to be corrected is too large, then increasing iterations might be advisable.

        TODO: include a vertical offset for a given ROI.
        """

        cfg = read_config_file(path=path)

        self.processing_dir = cfg["processing_dir"]
        self.datasets = cfg["datasets"]
        self.dering = False
        if cfg["dering"] == "True":
            self.dering = True

        self.overwrite = False
        if cfg["overwrite"] == "True":
            self.overwrite = True
        self.iterations = iterations
        self.slab_size = slab_size
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

    def _dering(self, path: str):
        t0 = time.time()
        if self.dering:
            with h5py.File(path, "r") as hin:
                projs = hin["projections"][:]
            data_vwu = np.rollaxis(projs, 1, 0)
            tmp = np.zeros_like(data_vwu)
            del projs
            try:
                with concurrent.futures.ProcessPoolExecutor() as pool:
                    for ii, result in enumerate(pool.map(remove_all_stripe, data_vwu)):
                        tmp[ii] = result
                data_vwu = tmp.copy()
                del tmp
                t1 = time.time()
                print(f"Deringed all {data_vwu.shape[0]} sinograms in {t1 - t0} s.")
            except ValueError:
                print(
                    "Probably this dataset was already deringed, skipping to the next."
                )

            projs = np.rollaxis(data_vwu, 1, 0)

            with h5py.File(path, "a") as hout:
                del hout["projections"]
                hout["projections"] = projs

    def run_alignment(self, xprop=None):
        """Method to run the alignment for the selected datasets."""

        for proc_path in self.processing_paths:

            print(f"Will align {get_dataset_name(proc_path)}.")

            try:
                projs, angles_rad, is_aligned, is_return = self._load_data(
                    path=proc_path, xprop=xprop
                )
            except Exception as e:
                print(e)
                print("Tour scan is probably broken, going to the next.")
                continue

            if self.dering and not is_aligned:
                try:
                    self._dering(proc_path)
                except Exception as e:
                    print(e)
                    print("Something went wrong, continuing.")

            if is_return:
                self._update_reverse_scan(path=proc_path)

                try:
                    projs, angles_rad, is_aligned, is_return = self._load_data(
                        path=proc_path, xprop=xprop
                )
                except Exception as e:
                    print(e)
                    print("Tour scan is probably broken, going to the next.")
                    continue

            if not is_aligned:
                projs_bin = binning(projs)

                del projs

                # ang0 = angles[0]
                # angles_rad = np.deg2rad(angles) - np.deg2rad(ang0)

                data_vwu = np.rollaxis(projs_bin, 1, 0)

                optim = cct.utils_align.OptimizeDetectorShifts(
                    data_vwu,
                    angles_rad,
                    solver_cls=cct.solvers.FBP,
                    solver_opts={},
                    verbose=False,
                )

                try:
                    pre_shifts_v = optim.pre_align_shifts_v()
                    pre_shifts_u, cor = optim.pre_align_shifts_u(
                        background=0.1, robust=True
                    )

                    pre_shifts_vu = np.stack([pre_shifts_v, pre_shifts_u + cor], axis=0)
                    print(pre_shifts_vu)
                except Exception as e:
                    print(e)
                    continue

                cor2 = optim.pre_cor_u_360()
                print(f"Center-of-rotation found using 360 redundancy: {cor2}")

                shifts, _ = optim.tomo_consistency_traditional(
                    cor2, iterations=self.iterations
                )

                print(shifts)

                self._save_shifts(path=proc_path, shifts=2 * shifts, cor=2 * cor2)

            else:
                if self.overwrite:
                    projs_bin = binning(projs)

                    del projs

                    data_vwu = np.rollaxis(projs_bin, 1, 0)

                    optim = cct.utils_align.OptimizeDetectorShifts(
                        data_vwu,
                        angles_rad,
                        solver_cls=cct.solvers.FBP,
                        solver_opts={},
                        verbose=False,
                    )

                    pre_shifts_v = optim.pre_align_shifts_v()
                    pre_shifts_u, cor = optim.pre_align_shifts_u(
                        background=0.1, robust=True
                    )

                    pre_shifts_vu = np.stack([pre_shifts_v, pre_shifts_u + cor], axis=0)
                    print(pre_shifts_vu)

                    cor2 = optim.pre_cor_u_360()
                    print(f"Center-of-rotation found using 360 redundancy: {cor2}")

                    shifts, _ = optim.tomo_consistency_traditional(
                        cor2, iterations=self.iterations
                    )

                    self._save_shifts(path=proc_path, shifts=2 * shifts, cor=2 * cor2)
                else:
                    print(
                        f"{get_dataset_name(proc_path)} is already aligned, skipping."
                    )

    def _load_data(self, path: str, xprop=None):
        is_aligned = False
        is_return = False
        with h5py.File(path, "r") as hin:
            if "shifts" in hin.keys():
                is_aligned = True
            nz, ny, nx = hin["projections"].shape
            ymin = (ny // 2) - (self.slab_size // 2)
            ymax = (ny // 2) + (self.slab_size // 2)
            if xprop is None:
                projs = hin["projections"][:, ymin:ymax, :].astype(np.float32)
            else:
                xmin = int((nx // 2) - np.ceil(xprop * nx))
                xmax = int((nx // 2) + np.ceil(xprop * nx))
                if xmin % 2 != 0:
                    xmin -= 1
                if xmax % 2 != 0:
                    xmax += 1

                print("xmin and xmax are", xmin, xmax)
                projs = hin["projections"][:, ymin:ymax, xmin:xmax].astype(np.float32)
            angles = hin["angles"][:]

        if self._is_return_scan(angles=angles):
            is_return = True
            projs = np.flip(projs, axis=(0, 1))
            angles = np.flip(angles, axis=0)

        return projs, np.deg2rad(angles), is_aligned, is_return

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
            return True
        else:
            return False

    def _save_shifts(
        self,
        path: str,
        shifts: np.ndarray,
        cor: float,
    ):
        with h5py.File(path, "a") as hout:
            if not ("cor" in hout.keys()):
                hout["cor"] = cor
            else:
                hout["cor"][...] = cor
            if not ("shifts" in hout.keys()):
                hout["shifts"] = shifts
            else:
                hout["shifts"][...] = shifts

    def _update_reverse_scan(self, path: str):
        with h5py.File(path, "a") as hout:
            projs = hout["projections"][:]
            angles = hout["angles"][:]
            tmp_prj = projs.copy()
            tmp_ang = angles.copy()
            hout["projections"][...] = np.flip(tmp_prj, axis = (0, 1))
            hout["angles"][...] = np.flip(tmp_ang, axis=0)
            print("Updated reverse scan !")
