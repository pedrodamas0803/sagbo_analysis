import os

import corrct as cct
import h5py
import numpy as np

from .reconstruction_utils import get_dataset_name, read_config_file


class Reconstruction:

    def __init__(self, path: str, increment: int = 1, sirt_iter: int = 0):

        cfg = read_config_file(path)
        self.datasets = cfg['datasets']
        self.processing_dir = cfg['processing_dir']
        self.increment = increment
        self.sirt_iter = sirt_iter

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

    def run_reconstruction(self):

        solverFBP = cct.solvers.FBP(verbose=False, fbp_filter='hann')

        for dataset in self.processing_paths:

            data_vwu, angles_rad, shifts, volFBP = self._load_data(dataset)

            init_angle = angles_rad[0]

            angles_rad = angles_rad - init_angle

            vol_geom = cct.models.VolumeGeometry.get_default_from_data(
                data_vwu)
            proj_geom = cct.models.ProjectionGeometry.get_default_parallel()
            proj_geom.set_detector_shifts_vu(shifts)

            if volFBP is not None:

                print('Found FBP volume, will use it as initial guess for SIRT.')

                with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, prj_geom=proj_geom) as A:
                    volFBP, _ = solverFBP(A, data_vwu, iterations=100)

                with h5py.File(dataset, 'a') as hout:
                    if 'volFBP' in hout.keys():
                        del hout['volFBP']
                    hout['volFBP'] = volFBP
                print('Reconstructed FBP volume and wrote it to file.')

            if self.sirt_iter > 0:

                solverSIRT = cct.solvers.Sirt(verbose=False)
                with cct.projectors.ProjectorUncorrected(vol_geom, angles_rad, prj_geom=proj_geom) as A:
                    volSIRT, _ = solverSIRT(
                        A, data_vwu, x0=volFBP, iterations=self.sirt_iter)

                with h5py.File(dataset, 'a') as hout:
                    if 'volSIRT' in hout.keys():
                        del hout['volSIRT']
                    hout['volSIRT'] = volSIRT
                print('Reconstructed SIRT volume and wrote it to file.')

                del volSIRT, solverSIRT

            del data_vwu, angles_rad, shifts, volFBP, vol_geom, proj_geom

    def _load_data(self, path: str):

        with h5py.File(path, 'r') as hin:
            x0 = None
            if 'volFBP' in hin.keys():
                x0 = hin['volFBP']

            projs = hin['projections'][:].astype(np.float32)
            angles = hin['angles'][:]
            shifts = hin['shifts'][:]

        return np.rollaxis(projs, 1, 0), np.deg2rad(angles), shifts, x0

    # def _check_FBP_rec(self, path:str):
    #     has_FBP = False
    #     with h5py.File(path, 'r') as hin:
    #         if 'volFBP' in hin.keys():
    #             has_FBP = True
    #     return has_FBP
