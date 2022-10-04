from curses.ascii import FF
import os
import concurrent.futures
from statistics import mean

import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from .utils import dezinger, ccij

NTHREAD = os.cpu_count() - 1

# 
# def zinger_remove( img, reference, medsize=3, nsigma=5 ):
#     '''
#     remove zingers. Anything which is >5 sigma after a 3x3 median filter is replaced by the filtered values
#     '''
#     dimg = img - reference
#     med = ndi.median_filter( dimg, medsize )
#     err = dimg - med
#     ds0 = err.std()
#     msk = err > ds0*nsigma
#     gromsk = ndi.binary_dilation( msk )
#     return np.where( gromsk, med, img )

        
# def dezinger( in_imgs ):
#     '''
#     Everything should work well if you define a median_flat as a variable 
#     '''
#     t0 = time.time()
#     flats, darks = in_imgs
#     N = flats.shape[0]
#     median_dark = np.median(darks, axis = 0)
#     del darks

#     with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
#         def work(i):
#             return i,zinger_remove( flats[i], median_dark )
#         for i, result in pool.map( work, range(len(flats)) ):
#             flats[i] = result
#     t1 = time.time()

#     print(f'It took {(t1-t0):.2f}s to dezinger {N} images.')
#     return flats

# def ccij(args):
#     """  
#     Compute (img[i]*img[j]).sum() / npixels
#     wrapper to be able to call via threads
#     args == i, j, npixels, imgs
#     """ 
#     i,j,NY,imgs = args
#     return i,j,np.einsum( 'ij,ij', imgs[i], imgs[j] ) / NY

class BasePCAFlat:

    def __init__(self):

        self.flats = np.empty(shape=(2048,2048), dtype=np.float32)
        self._darks = np.empty(shape=(2048,2048), dtype=np.float32)
        self.projections = np.empty(shape=(2048,2048), dtype=np.float32)
        self.mask = np.ones(shape=(2048,2048), dtype=bool)
        self.components = np.empty(shape=(2048,2048), dtype=np.float32)
        self.dark = np.empty(shape=(2048,2048), dtype=np.float32)
        self.mean = np.empty(shape=(2048,2048), dtype=np.float32)

    
    
    def setmask(self, prop = 0.125):
        lim = int(prop * self.mean.shape[1])
        self.mask = np.zeros(self.mean.shape, dtype = bool)
        self.mask[:, :lim ] = True
        self.mask[:, -lim:] = True
        self.g = [np.ones(self.mask.sum())] + [ component[self.mask] for component in self.components]

    def update_mask(self, mask:np.ndarray):
        if mask.dtype == bool:
            self.mask = mask
        else:
            print('Wrong type of mask, keeping initial mask.')

    def setdark(self):
        self.dark = np.median(self._darks, axis = 0)

    def _set_projections(self, projections:np.ndarray):
        
        projections = dezinger((projections, self.dark))
        self.projections = projections

    def correctproj(self, projection):
        logp = np.log( projection.astype( np.float32 ) - self.dark )
       
        cor = self.mean - logp
        # model to be fitted !!
        return self.fit( cor )
    
    def readcorrect1(self, ii):
        cor, s =  self.correctproj( self.projections[ii] )
        return ii, cor, s
    
    
    def fit(self, cor ):
        """ This is for each projection, so worth optimising ... """
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

        with h5py.File(path, "w") as hout:
            # solution = hout.create_dataset( "fitvals", shape = (len(projections), len(p.g)), dtype=float )
            # projections = hout.create_dataset( "projections", shape = projections.shape, dtype = np.float32,
            #                                 chunks = (1, projections.shape[1], projections.shape[2] ) )
            hout['dark'] = self.dark
            hout['p_mean'] = self.mean
            hout['p_components'] = np.array(self.components)
            hout['p_mask'] = self.mask
            hout.flush()


class PCAFlatImages(BasePCAFlat):

    """

    """
    def __init__(self, flats: np.ndarray, darks:np.ndarray = None, projections:np.ndarray = None):

        """
        Inputs: 
        Flats: np.ndarray; a stack of dark corrected flat field images
        Darks: np.ndarray; an image or stack of images of the dark current images of the camera.
        Projections: np.ndarray; stack of radiographs to be used to calculate 

        Does the log scaling.
        Subtracts mean and does eigenvector decomposition
        """
        super().__init__()
        self.flats = self._allocate_memory(flats.shape)  # take a log here

        if darks is not None:
            self._darks = darks.astype(np.float32)
        else:
            self._darks = np.zeros(flats[0].shape, dtype = np.float32)

        self._prepare_flats(flats, darks)

        self.projections = projections
        self.mean = np.mean(self.flats, axis = 0)  # average
        self.mask = self.__calculate_mask()        
        self.g = []

        self.flats = self.flats - self.mean # makes a copy
        self.cov = self.correlate( )
        self.decompose()

    def __calculate_mask(self, mult=20):

        if self.projections is not None:

            mean_x = self.projections.mean(axis=(0,1))
            thrs = mean_x.mean() + mult * mean_x.std()

            def _calc_ii(mean_x):
                for ii, value in enumerate(mean_x):
                    if value > thrs:
                        return ii
            ii = _calc_ii(mean_x)

            self.mask = np.zeros(self.projections[0].shape, dtype=bool)
            self.mask[:, :ii] = True
            self.mask[:, -ii:]
        
        else:

            self.mask = np.ones(self.flats[0].shape, dtype=bool)

        return self.mask
                
    
    def _allocate_memory(self, shape:tuple, dtype: np.dtype = np.float32):

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
        
        def calc1(i):
            calc = np.einsum('i,ijk->jk', self.eigenvectors[:,i], self.flats) 
            norm = (calc**2).sum()
            return i, calc / np.sqrt( norm )
        
        with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
            for i, result in pool.map( calc1, range(N) ):
                self.components[i] = result
        # simple gradients
        r, c = self.components[0].shape
        
        self.components.append(  np.outer( np.ones( r ), np.linspace(-1/c,1/c,c) ) )
        self.components.append(  np.outer( np.linspace(-1/r,1/r,r), np.ones(c) ) )
        
                               
            
    def plotComponents(self):
        N = len(self.components)
        row = int( np.floor( np.sqrt(N+2) ) )
        col = int( np.ceil( (N+2) / row ) )
        f, ax = plt.subplots( row, col, figsize = (16, 8))
        a = ax.ravel()
        a[0].plot( self.eigenvalues[:-1], "+-" )
        a[0].plot( self.eigenvalues[:N], "o-")
        a[0].set(yscale='log', title='eigenvalues')
        img = a[1].imshow( self.mean )
        a[1].set(title='mean')
        f.colorbar( img, ax=a[1] )
        for i in range( N ):
            img = a[i+2].imshow( self.components[i] )
            a[i+2].set(title=str(i))
            f.colorbar( img, ax=a[i+2] )
        plt.show(False)
            
    def setmask(self):
        # self.mask = mask
        self.g = [np.ones(self.mask.sum())] + [ component[self.mask] for component in self.components ] 
    
    def setdark(self):
        self.dark = np.median(self._darks, axis = 0)
            
    # def correctproj(self, projection):
    #     logp = np.log( projection.astype( np.float32 ) - self.dark )
       
    #     cor = self.mean - logp
    #     # model to be fitted !!
    #     return self.fit( cor )
        
    # def fit(self, cor ):
    #     """ This is for each projection, so worth optimising ... """
    #     y = cor[self.mask]
    #     g = self.g   # gradients
    #     # Form RHS
    #     rhs = [ np.dot( y, g[i]) for i in range(len(g)) ]
    #     # Form LSQ matrix
    #     mat = np.zeros( (len(g), len(g)), float )
    #     for i in range( len(g) ):
    #         mat[i,i] = np.dot(g[i],g[i])
    #         for j in range(i):
    #             mat[i,j] = mat[j,i] = np.dot(g[i],g[j])
    #     # Solve
    #     pinv = np.linalg.pinv( mat, hermitian=True )
    #     solution = np.dot( pinv, rhs )
    #     # Compute
    #     calc = np.full( self.components[0].shape, solution[0] )
    #     for i in range(len(g)-1):
    #         np.add( calc, self.components[i] * solution[i+1] , calc )
    #     return cor - calc, solution

    # def save_decomposition(self, path:str):

    #     with h5py.File(path, "w") as hout:
    #         # solution = hout.create_dataset( "fitvals", shape = (len(projections), len(p.g)), dtype=float )
    #         # projections = hout.create_dataset( "projections", shape = projections.shape, dtype = np.float32,
    #         #                                 chunks = (1, projections.shape[1], projections.shape[2] ) )
    #         hout['dark'] = self.dark.astype(np.float32)
    #         hout['p_mean'] = self.mean.astype(np.float32)
    #         hout['p_components'] = np.array( self.components ).astype(np.float32)
    #         hout['p_mask'] = self.mask
    #         hout.flush()

class PCAFlatFromFile(BasePCAFlat):

    """
    
    Class to read the results of the flat field flat field decomposition from a saved file using PCAFlatImages.

    Ideally the decomposition could be placed in the acquisition directory and the corrected projections saved inside each dataset folder.

    """


 

    def __init__(self, path:str):

        """
        
        Inputs
        - path: str; string with the path to the saved PCA decomposition file.

        """
        super().__init__()

        self.path = path
        self._read_PCA_flat_from_file()
        # self.__calculate_mask()

    def _read_PCA_flat_from_file(self):

        with h5py.File(self.path, 'r') as hin:

            self._darks = hin['dark'][:].astype(np.float32)
            self.dark = hin['dark'][:].astype(np.float32)
            self.mean = hin['p_mean'][:].astype(np.float32)
            self.components = hin['p_components'][:].astype(np.float32)
            self.mask = hin['p_mask'][:].astype(np.float32)
        
        print("read the file content! ")

        

    








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


