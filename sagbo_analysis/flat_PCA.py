import os
import concurrent.futures

import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

NTHREAD = os.cpu_count()-1

# 
def zinger_remove( img, reference, medsize=3, nsigma=5 ):
    '''
    remove zingers. Anything which is >5 sigma after a 3x3 median filter is replaced by the filtered values
    '''
    dimg = img - reference
    med = ndi.median_filter( dimg, medsize )
    err = dimg - med
    ds0 = err.std()
    msk = err > ds0*nsigma
    gromsk = ndi.binary_dilation( msk )
    return np.where( gromsk, med, img )

        
def dezinger( in_imgs ):
    '''
    Everything should work well if you define a median_flat as a variable 
    '''

    flats, darks = in_imgs
    median_dark = np.median(darks, axis = 0)
    del darks

    with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
        def work(i):
            return i,zinger_remove( flats[i], median_dark )
        for i, result in pool.map( work, range(len(flats)) ):
            flats[i] = result
    return flats

def ccij(args):
    """  
    Compute (img[i]*img[j]).sum() / npixels
    wrapper to be able to call via threads
    args == i, j, npixels, imgs
    """ 
    i,j,NY,imgs = args
    return i,j,np.einsum( 'ij,ij', imgs[i], imgs[j] ) / NY

class FlatPCA:
    def __init__(self, frames):
        """ 
        Frames is a stack of dark corrected flats
        Does the log scaling.
        Subtracts mean and does eigenvector decomposition
        """
        self.frames = np.empty( frames.shape, np.float32 )  # take a log here
        with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
            for i,frame in enumerate( pool.map( np.log, frames ) ):
                self.frames[i] = frame
        self.mean = np.mean( self.frames, axis = 0)  # average
        self.frames = self.frames - self.mean # makes a copy
        self.cov = self.correlate( )
        self.decompose()
        
    def correlate(self):
        """ 
        Computes an (nframes x nframes) correlation matrix
        """
        N = len(self.frames)
        CC = np.zeros( (N, N), float )
        args = [(i,j,N,self.frames) for i in range(N) for j in range(i+1)]
        with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
            for i,j,result in pool.map( ccij, args ):
                CC[i,j] = CC[j,i] = result
        return CC
        
    def decompose(self):
        """ Gets eigenvectors and eigenvalues and sorts them into order """
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.cov)
        order = np.argsort( abs(self.eigenvalues) )[::-1] # high to low
        self.eigenvalues = self.eigenvalues[order]
        self.eigenvectors = self.eigenvectors[:,order]
        
    def makeframes(self, nsigma=4):
        """ Projects the eigenvectors back into image space """
        av = abs( self.eigenvalues )
        N = (av > (av[-2]*nsigma)).sum()   # Go for 3 sigma
        print("Created",N,"components at", nsigma,"sigma")
        self.components = [None,] * N
        
        def calc1(i):
            calc = np.einsum('i,ijk->jk', self.eigenvectors[:,i], self.frames) 
            norm = (calc**2).sum()
            return i, calc / np.sqrt( norm )
        
        with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
            for i, result in pool.map( calc1, range(N) ):
                self.components[i] = result
        # simple gradients
        r,c=self.components[0].shape
        
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
            
    def setmask(self, mask):
        self.mask = mask
        self.g = [np.ones(mask.sum())] + [ c[mask] for c in self.components ] 
    
    def setdark(self, dark):
        self.dark = dark
            
    def correctproj(self, projection):
        logp = np.log( projection.astype( np.float32 ) - self.dark )
        # -log( projection / flat ) = log(flat) - log( projection )
        cor = self.mean - logp
        # model to be fitted !!
        return self.fit( cor )
        
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

    def save_decomposition(self, path:str):

        with h5py.File(path, "w") as hout:
            # solution = hout.create_dataset( "fitvals", shape = (len(projs), len(p.g)), dtype=float )
            # projections = hout.create_dataset( "projections", shape = projs.shape, dtype = np.float32,
            #                                 chunks = (1, projs.shape[1], projs.shape[2] ) )
            hout['dark'] = self.dark.astype(np.float32)
            hout['p_mean'] = self.mean.astype(np.float32)
            hout['p_components'] = np.array( self.components ).astype(np.float32)
            hout['p_mask'] = self.mask
            hout.flush()
