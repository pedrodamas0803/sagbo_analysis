import os
import numpy as np
import h5py
from skimage.exposure import rescale_intensity

from .archive_utils import read_config_file, get_dataset_name

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
    
    def reduce_mem_footprint(self, only_selected:bool = True, out_dtype:str = np.uint8):

        if out_dtype not in [np.uint8, np.uint16, np.uint32]:
            print('Unsuported out dtype, try unsigned int 8, 16 or 32 bits.')
            return 1

        if not only_selected:
            self.increment = 1

        for proc_path in self.processing_paths:
            try:
                vol = self._load_32bit_vol(path = proc_path)
                old_dtype = vol.dtype
            except Exception as e:
                print(e)
                continue
        
            resc_vol = rescale_intensity(vol, in_range=(vol.min(), vol.max()), out_range=np.uint8)
            print(f'Rescaled vol dtype: {resc_vol.dtype}')
            self._save_rescaled_vol(path=proc_path, vol=resc_vol)

            print(f'Changed {proc_path} FBP volume from {old_dtype} to {resc_vol.dtype}.')      
        

    def _load_32bit_vol(self, path:str):
        '''
        For the moment it only deals with volFBP !
        '''
        with h5py.File(path, 'r') as hin:
            if 'volFBP' in hin.keys():
                vol = hin['volFBP'][:]
        return vol
    
    def _save_rescaled_vol(self, path:str, vol:np.ndarray):

        '''
        For the moment it only deals with volFBP !
        '''
        with h5py.File(path, 'a') as hout:
            if 'volFBP' in hout.keys():
                del hout['volFBP']
            hout.create_dataset(name='volFBP', shape=vol.shape, dtype=vol.dtype, data=vol)
            # hout['volFBP'] = vol
            


    


