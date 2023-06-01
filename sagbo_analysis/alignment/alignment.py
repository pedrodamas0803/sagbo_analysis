import os
import concurrent.futures
import time

from algotom.prep.removal import remove_all_stripe
import corrct as cct
import h5py
import numpy as np

from .align_utils import binning, get_dataset_name, read_config_file


class ProjectionAlignment:

    ''' Class to perform projection alignment based on tomographic consistency.'''

    def __init__(self, path: str, increment: int = 1, iterations=5, slab_size=400):
        '''
        Inputs: 

        path: str -  the path to the configuration file where the data processing information is stored.
        increment: int - the step between each dataset to be processed among the time-series.
        iterations: int - number of iterations for the tomographic consistency alignment. Higher tends to be better, but takes longer.
        slab_size: int - number of slices, centered in the height of the projections, to be used for the alignment. Higher tends to be better, the data will be binned for the calculationsfor the calculations, don't choose a number too small.

        The default values tend to work well, unless the error-motion to be corrected is too large, then increasing iterations might be advisable.

        TODO: include a vertical offset for a given ROI.
        '''

        cfg = read_config_file(path=path)

        self.processing_dir = cfg['processing_dir']
        self.datasets = cfg['datasets']
        self.dering = cfg['dering']
        self.overwrite = False
        if cfg['overwrite'] == 'True':
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
            path_to_process = os.path.join(
                self.processing_dir, name, f'{name}.h5')
            proc_paths.append(path_to_process)
        return proc_paths

    def _dering(self, path:str):
        t0 = time.time()
        if self.dering:
            
            with h5py.File(path, 'r') as hin:
                projs = hin['projections'][:]
            data_vwu = np.rollaxis(projs, 1, 0)
            tmp = np.zeros_like(data_vwu)
            del projs
            try:
                with concurrent.futures.ProcessPoolExecutor() as pool:
                    for ii, result in enumerate(pool.map(remove_all_stripe, data_vwu)):
                        tmp[ii]= result
                data_vwu = tmp.copy()
                del tmp
                t1 = time.time()
                print(f'Deringed all {data_vwu.shape[0]} sinograms in {t1 - t0} s.')
            except ValueError:
                print("Probably this dataset was already deringed, skipping to the next.")         
            
            projs = np.rollaxis(data_vwu, 1, 0)

            with h5py.File(path, 'a') as hout:
                del hout['projections']
                hout['projections'] = projs
        

    
    def run_alignment(self, xprop = None):
        ''' Method to run the alignment for the selected datasets.'''

        for proc_path in self.processing_paths:

            self._dering(proc_path)

            print(f'Will align {get_dataset_name(proc_path)}.')

            projs, angles, is_aligned = self._load_data(path=proc_path, xprop=xprop)

            # dirty fix
            # is_aligned = False
            
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
                if self.overwrite:
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

    def _load_data(self, path: str, xprop = None):
        
        is_aligned = False
        with h5py.File(path, 'r') as hin:

            if 'shifts' in hin.keys() and 'cor' in hin.keys():
                is_aligned = True
                return None, None, is_aligned
            else:
                nz, ny, nx = hin['projections'].shape
                ymin = (ny // 2) - (self.slab_size//2)
                ymax = (ny // 2) + (self.slab_size//2)
                if xprop is None:
                    projs = hin['projections'][:, ymin:ymax, :].astype(np.float32)
                else:
                    xmin = int((nx // 2) - np.ceil(xprop * nx))
                    xmax = int((nx // 2) + np.ceil(xprop * nx))
                    if xmin % 2 != 0:
                        xmin -= 1
                    if xmax % 2 != 0:
                        xmax += 1
                    
                    print('xmin and xmax are', xmin, xmax)
                    projs = hin['projections'][:, ymin:ymax, xmin:xmax].astype(np.float32)
                angles = hin['angles'][:]

                return projs, angles, is_aligned
