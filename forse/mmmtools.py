import numpy as np
import healpy as hp
import reproject
from astropy.wcs import WCS
import astropy.io.fits as fits

def h2f(hmap,target_header,coord_in='G'):
    #project healpix -> flatsky
    pr,footprint = reproject.reproject_from_healpix(
    (hmap, coord_in), target_header, shape_out=(500,500),
    order='nearest-neighbor', nested=False)
    return pr

def f2h(flat,target_header,nside,coord_in='G'):
    #project flatsky->healpix
    pr,footprint = reproject.reproject_to_healpix(
    (flat, target_header),coord_system_out='G', nside=nside ,
    order='nearest-neighbor', nested=False)
    return pr

def set_header(ra,dec, size_patch , Npix=128):
    hdr = fits.Header()
    hdr.set('SIMPLE' , 'T')
    hdr.set('BITPIX' , -32)
    hdr.set('NAXIS'  ,  2)
    hdr.set('NAXIS1' ,  Npix)
    hdr.set('NAXIS2' ,  Npix )
    hdr.set('CRVAL1' ,  ra)
    hdr.set('CRVAL2' ,  dec)
    hdr.set('CRPIX1' ,  Npix/2. +.5)
    hdr.set('CRPIX2' ,  Npix/2. +.5 )
    hdr.set('CD1_1'  , size_patch )
    hdr.set('CD2_2'  , -size_patch )
    hdr.set('CD2_1'  ,  0.0000000)
    hdr.set('CD1_2'  , -0.0000000)
    hdr.set('CTYPE1'  , 'RA---ZEA')
    hdr.set('CTYPE2'  , 'DEC--ZEA')
    hdr.set('CUNIT1'  , 'deg')
    hdr.set('CUNIT2'  , 'deg')
    hdr.set('COORDSYS','icrs')
    return hdr

def make_train_set(Ntrain, m_hres, m_lres, Npix, patch_dim, seed=None):
    high_res_patches = []
    low_res_patches = []
    reso_amin = patch_dim*60./Npix
    sizepatch = reso_amin/60.
    if seed:
        np.random.seed(seed)
    for N in range(Ntrain):
        lat = np.random.uniform(-90,90)
        lon = np.random.uniform(0,360)
        header = set_header(lon, lat, sizepatch, Npix)
        if len(m_hres)>3:
            high_res_patches.append(h2f(m_hres, header))
            low_res_patches.append(h2f(m_lres, header))
        else:
            high_res_patch_TQU = np.zeros((len(m_hres), Npix, Npix))
            low_res_patch_TQU = np.zeros((len(m_lres), Npix, Npix))
            for i in range(len(m_hres)):
                high_res_patch_TQU[i] = h2f(m_hres[i], header)
                low_res_patch_TQU[i] = h2f(m_lres[i], header)
            high_res_patches.append(high_res_patch_TQU)
            low_res_patches.append(low_res_patch_TQU)
    patches = np.array([high_res_patches, low_res_patches])
    return patches

def split_training_set(xraw):
    nstamps = xraw.shape[-1]
    npix = xraw.shape[0]
    nchans = 1
    ntrains = int(nstamps *  4./5.)
    ntests = int(nstamps * 1./5.)
    train = xraw[:,:,:ntrains].T.reshape(ntrains,npix,npix, 1)
    test = xraw[:,:,-ntests:].T.reshape(ntests,npix,npix,  1)
    return train, test

def divide_image(image):
    test_set_zoom = np.zeros((25, 64, 64))
    ind = 0
    for xax in range(5):
        for yax in range(5):
            test_set_zoom[ind, :, :] = image[64*xax:64*(xax+1), 64*yax:64*(yax+1)]
            ind = ind+1
    return test_set_zoom

def unify_image(images):
    one_big = np.zeros((320, 320))
    ind = 0
    for xax in range(5):
        for yax in range(5):
            one_big[64*xax:64*(xax+1), 64*yax:64*(yax+1)] = images[ind, :, :, 0]
            ind = ind+1
    return one_big

def bin_history(history, bins=100):
    len_data = len(history)
    x = np.arange(len_data)+1
    num_bin = len_data//bins
    data_binned = []
    x_binned = []
    for i in range(num_bin):
        data_binned.append(np.mean(history[i*bins:(i+1)*bins]))
        x_binned.append(np.mean(x[i*bins:(i+1)*bins]))
    data_binned = np.array(data_binned)
    x_binned = np.array(x_binned)
    return x_binned, data_binned


def MinMaxRescale(x,a=0,b=1):
    xresc = (b-a)*(x- x.min() )/(x.max() - x.min() ) +a
    return xresc
