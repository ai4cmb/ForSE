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


def h2f(hmap, target_header, coord_in='C'):
    ## Imported from PICASSO https://github.com/giuspugl/picasso
    #project healpix -> flatsky
    pr, footprint = reproject.reproject_from_healpix(
        (hmap, coord_in), target_header, shape_out=(500,500),
        order='nearest-neighbor', nested=False)
    return pr

def f2h(flat, target_header, nside, coord_in='C'):
    ## Imported from PICASSO https://github.com/giuspugl/picasso
    #project flatsky->healpix
    pr, footprint = reproject.reproject_to_healpix(
        (flat, target_header),coord_system_out='C', nside=nside ,
        order='nearest-neighbor', nested=False)
    return pr

def set_header(ra,dec, size_patch , Npix=128):
    ## Imported from PICASSO https://github.com/giuspugl/picasso
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

def make_patches_from_healpix(
        Npatches, m_hres, m_lres, Npix, patch_dim, lat_lim=None, seed=None, mask=None):
    high_res_patches = []
    low_res_patches = []
    reso_amin = patch_dim*60./Npix
    sizepatch = reso_amin/60.
    if seed:
        np.random.seed(seed)
    if np.any(mask)==None:
        mask_hp = m_hres*0.+1
    else:
        mask_hp = mask
    for N in range(Npatches):
        if lat_lim:
            lat = np.random.uniform(-lat_lim,lat_lim)
        else:
            lat = np.random.uniform(-90,90)
        lon = np.random.uniform(0,360)
        header = set_header(lon, lat, sizepatch, Npix)
        mask_patch = h2f(mask_hp, header)
        if len(np.where(mask_patch>0)[0])/(Npix*Npix)>0.9:
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
