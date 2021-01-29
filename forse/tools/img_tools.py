import numpy as np
import healpy as hp
import reproject
from astropy.wcs import WCS
import astropy.io.fits as fits

def rescale_min_max(img, a=-1, b=1, return_min_max=False):
    img_resc = (b-a)*(img-np.min(img))/(np.max(img)-np.min(img))+a
    if return_min_max:
        return img_resc, np.min(img), np.max(img)
    else:
        return img_resc

def rescale_min_max_back(img, min_max):
    img_back = (img+1)/2.*(min_max[1]-min_max[0])+min_max[0]
    return(img_back)
