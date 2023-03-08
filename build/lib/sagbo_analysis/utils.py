
import concurrent.futures
import os
import time

import numpy as np
import scipy.ndimage as ndi
from scipy.signal import find_peaks
from skimage.exposure import histogram

NTHREAD = os.cpu_count() - 1


def zinger_remove(img, reference, medsize=3, nsigma=5):
    '''
    remove zingers. Anything which is >5 sigma after a 3x3 median filter is replaced by the filtered values
    '''
    dimg = img - reference
    med = ndi.median_filter(dimg, medsize)
    err = dimg - med
    ds0 = err.std()
    msk = err > ds0*nsigma
    gromsk = ndi.binary_dilation(msk)
    return np.where(gromsk, med, img)


def dezinger(in_imgs):
    '''
    Performs parallelized zinger removal.

    Inputs
    in_imgs: tuple; tuple in the form (flats, darks).

    Written this way to have only one argument. Could be improved, but it works.
    '''
    t0 = time.time()
    flats, darks = in_imgs
    N = flats.shape[0]
    print(f'Will dezinger {N} images. Might take few seconds.')
    median_dark = np.median(darks, axis=0)
    del darks

    with concurrent.futures.ThreadPoolExecutor(NTHREAD) as pool:
        def work(i):
            return i, zinger_remove(flats[i], median_dark)
        for i, result in pool.map(work, range(len(flats))):
            flats[i] = result
    t1 = time.time()

    print(f'It took {(t1-t0):.2f}s to dezinger {N} images.')
    return flats


def ccij(args):
    """  
    Compute (img[i]*img[j]).sum() / npixels
    wrapper to be able to call via threads
    args == i, j, npixels, imgs
    """
    i, j, NY, imgs = args
    return i, j, np.einsum('ij,ij', imgs[i], imgs[j]) / NY


def find_peak_position(image, height=1e4, retrn_counts=False):
    '''
    Finds the position of the peaks in a histogram of an image.

    Inputs  
    image - the image to be used
    height - the height of what should be considered as peaks in the curve
    retrn_counts - if True, returns the frequency associated to that peak position

    Outputs
    peaks - list of the peak positions. If retrn_counts == True, return tuples containing the counts.  
    '''

    counts, bins = histogram(image)

    peaks, _ = find_peaks(counts, height=height)
    peak_pos = []
    for peak in peaks:
        peak_pos.append(bins[peak])
    if not retrn_counts:
        return peak_pos
    else:
        return [(bins[peak], counts[peak]) for peak in peaks]


def calc_color_lims(img, mult=3):
    '''
    Calculates the upper and lower limits to plot an image with centered value on the brightest peak of the histogram.

    Inputs
    image - the image!
    mult - stretching/shrinking factor that defines the amplitude of vmax-vmin by using peak +/- mult*std

    Outputs
    vmin, vmax - tuple with the lower and upper limits. 
    '''
    peaks = find_peak_position(img)
    vmin = peaks[-1] - mult * img.std()
    vmax = peaks[-1] + mult * img.std()

    return vmin, vmax
