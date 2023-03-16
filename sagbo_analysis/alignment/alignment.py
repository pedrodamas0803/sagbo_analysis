import os

import corrct as cct
import h5py
import numpy as np

from .align_utils import binning, get_dataset_name, read_config_file


class ProjectionAlignment:

    def __init__(self, path: str, increment: int = 1, iterations=5, slab_size=400):

        cfg = read_config_file(path=path)

        self.processing_dir = cfg['processing_dir']
        self.datasets = cfg['datasets']
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
            path_to_process = os.path.join(
                self.processing_dir, name, f'{name}.h5')
            proc_paths.append(path_to_process)
        return proc_paths

    def run_alignment(self):

        for proc_path in self.processing_paths:

            print(f'Will align {get_dataset_name(proc_path)}.')

            projs, angles, is_aligned = self._load_data(path=proc_path)

            if not is_aligned:
                projs_bin = binning(projs)

                del projs

                angles_rad = np.deg2rad(angles)

                data_vwu = np.rollaxis(projs_bin, 1, 0)

                optim = cct.utils_align.OptimizeDetectorShifts(
                    data_vwu, angles_rad, solver_cls=cct.solvers.FBP,
                    solver_opts={}, verbose=False)

                pre_shifts_v = optim.pre_align_shifts_v()
                pre_shifts_u, cor = optim.pre_align_shifts_u(
                    background=0.1, robust=True)

                pre_shifts_vu = np.stack(
                    [pre_shifts_v, pre_shifts_u + cor], axis=0)
                print(pre_shifts_vu)

                cor2 = optim.pre_cor_u_360()
                print(f"Center-of-rotation found using 360 redundancy: {cor2}")

                shifts, _ = optim.tomo_consistency_traditional(
                    cor2, iterations=self.iterations)

                with h5py.File(proc_path, 'a') as hout:
                    hout['cor'] = 2 * cor2
                    hout['shifts'] = 2 * shifts
            else:
                print(f'{get_dataset_name(proc_path)} is already aligned, skipping.')

    def _load_data(self, path: str):

        is_aligned = False
        with h5py.File(path, 'r') as hin:

            if 'shifts' in hin.keys() and 'cor' in hin.keys():
                is_aligned = True
                return None, None, is_aligned
            else:
                nz, ny, nx = hin['projections'].shape
                ymin = (ny // 2) - (self.slab_size//2)
                ymax = (ny // 2) + (self.slab_size//2)
                projs = hin['projections'][:, ymin:ymax, :].astype(np.float32)
                angles = hin['angles'][:]

                return projs, angles, is_aligned
