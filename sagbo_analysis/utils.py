
import concurrent.futures as cf 

import numpy as np

from skimage.exposure import histogram
from scipy.signal import find_peaks


def find_peak_position(image, height = 1e4, retrn_counts = False):
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

def calc_color_lims(img, mult = 3):

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