import os
import concurrent.futures

import numpy as np
import h5py
import matplotlib.pyplot as plt

from .utils import dezinger, ccij

NTHREAD = os.cpu_count() - 1

class BasePCAFlat:

    """
    Base class to run PCA decomposition on flat field images and to perform the correction on each of the projections at a given acquisition dataset.
    
    This implementation was proposed by Jailin C. et al in https://doi.org/10.1107/S1600577516015812.

    Code written by ID11 @ ESRF staff. 
    
    Jonathan Wright - Implementation based on research paper
    Pedro D. Resende - Added saving and loading from file capabilities 

    """

    def __init__(self):

        """
        Initializes all variables needed to perform the PCA decomposition / correction with empty values to be used by the child classes.

        """

        self.flats = np.empty(shape=(2048,2048), dtype=np.float32)
        self._darks = np.empty(shape=(2048,2048), dtype=np.float32)
        self.projections = np.empty(shape=(2048,2048), dtype=np.float32)
        self.mask = np.ones(shape=(2048,2048), dtype=bool)
        self.components = np.empty(shape=(2048,2048), dtype=np.float32)
        self.dark = np.empty(shape=(2048,2048), dtype=np.float32)
        self.mean = np.empty(shape=(2048,2048), dtype=np.float32)

    
    
    def setmask(self, prop = 0.125):

        '''
        Sets the mask to select where the model is going to be fitted.

        Input
        prop: float, 0.125 by default;
        
        By default it sets the strips on each side of the frame in the form:
            mask[:, lim:] = True
            mask[:, -lim:] = True
        Where lim = prop * flat.shape[1]

        If you need a custom mask, see update_mask() method.

        '''
        lim = int(prop * self.mean.shape[1])
        self.mask = np.zeros(self.mean.shape, dtype = bool)
        self.mask[:, :lim ] = True
        self.mask[:, -lim:] = True
        self.g = [np.ones(self.mask.sum())] + [ component[self.mask] for component in self.components]

    def update_mask(self, mask:np.ndarray):

        """
        Method to update the mask with a custom mask in the form of a boolean 2+D array.

        Input
        mask: np.ndarray, float; 
        
        It will set the mask, replacing the standard mask created with setmask().
        """
        if mask.dtype == bool:
            self.mask = mask
        else:
            print('Wrong type of mask, keeping initial mask.')

    def setdark(self):

        """
        Generates the median dark image from the input images, whether from file or dark frames.
        """
        self.dark = np.median(self._darks, axis = 0)

    def _set_projections(self, projections:np.ndarray):

        """
        Sets the projections where the model will be fitted as an attribute of the class.
        
        Input
        projections: np.ndarray, float; the stack of radiographs to be flat corrected

        It performs zinger removal prior to setting the projections attribute.
        """
        
        projections = dezinger((projections, self.dark))
        self.projections = projections

    def plotComponents(self):

        """
        Plots the components that were calculated by PCA decomposition.
        """

        N = len(self.components)
        row = int( np.floor( np.sqrt(N+2) ) )
        col = int( np.ceil( (N+2) / row ) )
        f, ax = plt.subplots( row, col, figsize = (16, 8))
        a = ax.ravel()
        img = a[0].imshow( self.mean )
        a[0].set(title='mean')
        f.colorbar( img, ax=a[0] )
        for i in range( N ):
            img = a[i+1].imshow( self.components[i] )
            a[i+1].set(title=str(i))
            f.colorbar( img, ax=a[i+1] )
        f.tight_layout()
        plt.show(False)


    def correctproj(self, projection):

        """
        Performs the correction on one projection of the stack.
        Input

        projection: np.ndarray, float; radiograph from the acquisition stack.
        """
        logp = np.log( projection.astype( np.float32 ) - self.dark )
       
        cor = self.mean - logp
        # model to be fitted !!
        return self.fit( cor )
    
    def readcorrect1(self, ii):

        """
        Method to allow correction in a paralelized manner.
        """
        cor, s =  self.correctproj( self.projections[ii] )
        return ii, cor, s
    
    
    def fit(self, cor):

        """ 
        Performs the fit to determine the weight coefficient of each component to correct a given projection. 

        This is for each projection, so worth optimising ... 
        """

        y = cor[self.mask]
        g = self.g   # gradients
        # Form RHS
        rhs = [ np.dot( y, g[i]) for i in range(len(g)) ]
        # Form LSQ matrix
        mat = np.zeros( (len(g), len(g)), float )
        for i in range( len(g) ):
            mat[i,i] = np.dot(g[i],g[i])
            for j in range(i):
                mat[i,j] = mat[j,i] = np.dot(g[i],g[j])
        # Solve
        pinv = np.linalg.pinv( mat, hermitian=True )
        solution = np.dot( pinv, rhs )
        # Compute
        calc = np.full( self.components[0].shape, solution[0] )
        for i in range(len(g)-1):
            np.add( calc, self.components[i] * solution[i+1] , calc )
        return cor - calc, solution

    def correct_stack(self, projections: np.ndarray, save_path:str):

        """
        Performs the flat field correction at each projection and saves it into a file.

        Inputs
        projections: np.ndarray, float; the radiographs stack
        save_path: str; full path to the output h5 file where you want to save it
        """

        self._set_projections(projections)
        # self.mask = self.__calculate_mask()
        self.setmask()
        self.setdark()


        with h5py.File(save_path, "w") as hout:
            solution = hout.create_dataset( "fitvals", shape = (len(projections), len(self.g)), dtype = float )
            projections = hout.create_dataset( "projections", shape = projections.shape, dtype = np.float32,
                                            chunks = (1, projections.shape[1], projections.shape[2] ) )
            hout['dark'] = self.dark.astype(np.float32)
            hout['p_mean'] = self.mean.astype(np.float32)
            hout['p_components'] = np.array(self.components).astype(np.float32)
            hout['p_mask'] = self.mask
            hout.flush()
            with concurrent.futures.ThreadPoolExecutor( NTHREAD) as pool: # -1 for self as writer
                for i, cor, sol in pool.map( self.readcorrect1, range(len(projections)) ):
                    solution[i] = sol
                    projections[i] = cor
                    hout.flush()
                    if i % NTHREAD == 0:
                        print('Done '+str(i))

    def save_decomposition(self, path:str):

        """
        Saves the basic information of a PCA decomposition allowing the correction to be loaded by PCAFlatFromFile.

        Input:
        path: str; full path to the h5 file you want to save your results. It will overwrite!! Be careful.

        """

        with h5py.File(path, "w") as hout:
            hout['eigenvalues'] = self.ei
            hout['dark'] = self.dark
            hout['p_mean'] = self.mean
            hout['p_components'] = np.array(self.components)
            hout['p_mask'] = self.mask
            hout.flush()


class PCAFlatImages(BasePCAFlat):

    """
    Class to perform the PCA decomposition from a stack of flats and a stack of darks. 
    The main purpose of this is to save a decomposition with representative flats from your experiment (or a subset), 
    and then reinitialize the object using PCAFlatFromFile and perform the corrections for each dataset. 

    """
    def __init__(self, flats: np.ndarray, darks:np.ndarray = None, projections:np.ndarray = None):

        """
        Inputs: 
        Flats: np.ndarray; a stack of dark corrected flat field images
        Darks: np.ndarray; an image or stack of images of the dark current images of the camera.
        Projections: np.ndarray, optional; stack of radiographs to be corrected. 

        If projections is not None, it will correct the projection stack used on the initialization of the object.

        Does the log scaling.
        Subtracts mean and does eigenvector decomposition.

        """
        super().__init__()
        self.flats = self._allocate_memory(flats.shape)  # take a log here

        if darks is not None:
            self._darks = darks.astype(np.float32)

        self._prepare_flats(flats, darks)

        self.projections = projections
        self.mean = np.mean(self.flats, axis = 0)  # average
        self.mask = self.__calculate_mask()        
        self.g = []

        self.flats = self.flats - self.mean # makes a copy
        self.cov = self.correlate( )
        self.decompose()
                
    
    def _allocate_memory(self, shape:tuple, dtype: np.dtype = np.float32):
        """
        Initializes a new numpy array with the required shape.

        Args:
            shape (tuple): shape of the array (M x N x L).
            dtype (np.dtype, optional): type. Defaults to np.float32.

        Returns:
            empty array of shape (shape)
        """
        return np.empty(shape, dtype=dtype)
    
    def _prepare_flats(self, flats:np.ndarray, darks:np.ndarray):

        """
        
        Dezinger and takes log of the flat stack
        
        """

        flats = dezinger((flats, darks))

        with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
            for i,frame in enumerate( pool.map( np.log, flats ) ):
                self.flats[i] = frame

        
    def correlate(self):
        
        """ 
        
        Computes an (nflats x nflats) correlation matrix
        
        """
        N = len(self.flats)
        CC = np.zeros( (N, N), float )
        args = [(i,j,N,self.flats) for i in range(N) for j in range(i+1)]
        with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
            for i,j,result in pool.map( ccij, args ):
                CC[i,j] = CC[j,i] = result
        return CC
        
    def decompose(self):
        
        """
        
        Gets eigenvectors and eigenvalues and sorts them into order 
        
        """
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.cov)
        order = np.argsort( abs(self.eigenvalues) )[::-1] # high to low
        self.eigenvalues = self.eigenvalues[order]
        self.eigenvectors = self.eigenvectors[:,order]
        
    def makeframes(self, nsigma=3):
        
        """
        
        Projects the eigenvectors back into image space 
        
        """
        av = abs( self.eigenvalues )
        N = (av > (av[-2]*nsigma)).sum()   # Go for 3 sigma
        print("Created",N,"components at", nsigma,"sigma")
        self.components = [None,] * N
        
        def calculate(ii):
            calc = np.einsum('i,ijk->jk', self.eigenvectors[:,ii], self.flats) 
            norm = (calc**2).sum()
            return ii, calc / np.sqrt( norm )
        
        with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
            for ii, result in pool.map( calculate, range(N) ):
                self.components[ii] = result
        # simple gradients
        r, c = self.components[0].shape
        
        self.components.append(  np.outer( np.ones( r ), np.linspace(-1/c,1/c,c) ) )
        self.components.append(  np.outer( np.linspace(-1/r,1/r,r), np.ones(c) ) )
        
                               
            
    # def plotComponents(self):
    #     N = len(self.components)
    #     row = int( np.floor( np.sqrt(N+2) ) )
    #     col = int( np.ceil( (N+2) / row ) )
    #     f, ax = plt.subplots( row, col, figsize = (16, 8))
    #     a = ax.ravel()
    #     a[0].plot( self.eigenvalues[:-1], "+-" )
    #     a[0].plot( self.eigenvalues[:N], "o-")
    #     a[0].set(yscale='log', title='eigenvalues')
    #     img = a[1].imshow( self.mean )
    #     a[1].set(title='mean')
    #     f.colorbar( img, ax=a[1] )
    #     for i in range( N ):
    #         img = a[i+2].imshow( self.components[i] )
    #         a[i+2].set(title=str(i))
    #         f.colorbar( img, ax=a[i+2] )
    #     plt.show(False)
            

class PCAFlatFromFile(BasePCAFlat):

    """
    
    Class to read the results of a PCA  decomposition from a saved file using PCAFlatImages.

    Ideally the decomposition could be placed in the acquisition directory and the corrected projections 
    saved inside each dataset folder.

    """

    def __init__(self, path:str):

        """
        
        Inputs
        - path: str; string with the path to the saved PCA decomposition file.

        """
        super().__init__()

        self.path = path
        self._read_PCA_flat_from_file()
       
    def _read_PCA_flat_from_file(self):

        with h5py.File(self.path, 'r') as hin:

            self._darks = hin['dark'][:].astype(np.float32)
            self.dark = hin['dark'][:].astype(np.float32)
            self.mean = hin['p_mean'][:].astype(np.float32)
            self.components = hin['p_components'][:].astype(np.float32)
            self.mask = hin['p_mask'][:].astype(np.float32)
        
        print("Read the file content!")

        

    








if __name__ == '__main__':
    
    pth = '/data/visitor/ihma298/id11/test.h5'
    pca = PCAFlatFromFile(path=pth)

    print(pca.mean)

    print('Will try to read content from file')

    pth = '/data/visitor/ihma298/id11/g5_s4/g5_s4_pct_RT_cracked_65N/PCA.h5'
    pca = PCAFlatFromFile(path=pth)

    print(pca.mean)
    # f, axs = plt.subplots()

    # axs.imshow(pca.mean)

    # plt.show()


    # pca.projections = 'This is a test'

    # print(pca.projections)

    # pca.save_decomposition('/data/visitor/ihma298/id11/test.h5')


