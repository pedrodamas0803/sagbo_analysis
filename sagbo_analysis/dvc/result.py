from dataclasses import dataclass

import numpy as np
import h5py

@dataclass
class DVC_result:
    
    def __init__(self, res_path:str, z_pix_offset:float, y_pix_offset:float, x_pix_offset:float):
        
        self.path = res_path
        self.z_offset = z_pix_offset
        self.y_offset = y_pix_offset
        self.x_offset = x_pix_offset

        with h5py.File(self.path, 'r') as hin:

            self.Nelems = hin['Nelems'][:]
            self.Nnodes = hin['Nnodes'][:]
            self.Smesh = hin['Smesh'][:]
            self.U = hin['U'][:]
            self.conn = hin['conn'][:]
            self.elt = hin['elt'][:]
            self.mesh_size = hin['model/mesh_size'][0,0]
            self.ng = hin['ng'][:]
            self.nmod = hin['nmod'][:]
            self.ns = hin['ns'][:]
            self.regularization_parameter = hin['param/regularization_parameter'][0,0]
            self.rint = hin['rint'][:]
            self.xo = hin['xo'][:]
            self.yo = hin['yo'][:]
            self.zo = hin['zo'][:]
            
    def _assemble_theoretical_U(self):
        
        Ux = np.ones(self.xo.T.shape)*self.x_offset
        Uy = np.ones(self.yo.T.shape)*self.y_offset
        Uz = np.ones(self.zo.T.shape)*self.z_offset
        
        return np.concatenate((Ux, Uy, Uz))
    
    def systematic_error(self, return_theoretical:bool = False): 
        
        """
        Calculates the systematic error between a known offset of a volume in each 
        direction and the ones determined by DVC for purposes of sensitivity determination.
        
        Equation 3.21 in the Joel Lachambre thesis.
        """
        
        
        theor_U = self._assemble_theoretical_U()
        
        std = (1 / len(self.U.T)) * np.sum((theor_U - self.U.T))
        
        if return_theoretical:
            return std, theor_U
        
        return std
    
    def std(self, return_theoretical:bool = False): 
        
        """
        Calculates the standard deviation between a known offset of a volume in each 
        direction and the ones determined by DVC for purposes of sensitivity determination.
        
        Equation 3.22 in the Joel Lachambre thesis.
        """
        
        theor_U = self._assemble_theoretical_U()
        
        std = np.sqrt((1 / len(self.U.T)) * np.sum((theor_U - self.U.T) ** 2))
        
        if return_theoretical:
            return std, theor_U
        
        return std
    
    
    
    def sigma_total(self):
        
        '''
        Incertitude calculation based on equation 3.24 of Joel Lachambre thesis.
        '''        
        
        Ux, Uy, Uz = np.split(self.U.T, 3)
        
        Uxt, Uyt, Uzt = np.split(self._assemble_theoretical_U(), 3)
        
        sigx = (1/Ux.shape[0]) * np.sum((Ux - Uxt) ** 2)
        sigy = (1/Uy.shape[0]) * np.sum((Uy - Uyt) ** 2)
        sigz = (1/Uz.shape[0]) * np.sum((Uz - Uzt) ** 2)
        
        sigma_tot =  np.sqrt((1/3) *(sigx + sigy + sigz))
        
        return sigma_tot