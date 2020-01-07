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
    # No need of this function , we can just use rescale_min_max 
    # with rescale_min_max( img,  a= img.min(), and b = img.max() ) 
    
    img_back = (img+1)/2.*(min_max[1]-min_max[0])+min_max[0]
    return(img_back)

## Imported from PICASSO https://github.com/giuspugl/picasso

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
        Npatches, m_hres, m_lres, Npix, patch_dim, lat_lim=None, seed=None):
    high_res_patches = []
    low_res_patches = []
    reso_amin = patch_dim*60./Npix
    sizepatch = reso_amin/60.
    if seed:
        np.random.seed(seed)
    for N in range(Npatches):
        if lat_lim:
            lat = np.random.uniform(-lat_lim,lat_lim)
        else:
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

def divide_image(image, step, side) :
    """
    Parameters
    ----------
    image : array-like object shape = ( npix, npix )
    step : lenght to move in decomposition
    side : side lenght of output images

    Returns
    -------
    image array, pixel weight, limiting indexes of images in array
    """
    xnpix, ynpix = image.shape
    if (xnpix==ynpix):
        imgside = xnpix
    else:
        raise ValueError("Function forward only works for square images.")
    if (step<=side):
        pass
    else:
        raise ValueError("'side' should be greater than or equal to 'step'.")
    if (side<=imgside ):
        pass
    else:
        raise ValueError("'side' should be strictly smaller than the original image side lenght.")
    if imgside%step==0:
        pass
    else:
        raise ValueError("Step lenght selected not valid ('imside' should be a multipole of 'step')")
    # create array of ranges for slicing
    img_index = np.array([[idx, min(idx+side, ynpix)]
                        for idx in range(0, imgside, step)])
    sider = np.array([np.arange(idx[0], idx[1]) for idx in img_index])
    # create output array of images
    if step==side:
        NN = int(imgside/step)
    else:
        NN = int(imgside/step)-1
    img_array = np.zeros(shape=(NN*NN, side, side))
    img_weights = np.zeros(shape=image.shape)
    for ii in range(0, NN):
        for jj in range(0, NN):
            img_array[ii * NN + jj,
                       :img_index[ii, 1] - img_index[ii, 0],
                       :img_index[jj, 1] - img_index[jj, 0]] += image[np.ix_(sider[ii], sider[jj])]
            img_weights[np.ix_(sider[ii], sider[jj])] += 1
    return img_array, img_weights, img_index

def unify_images(imgs_array):
    Nimg = imgs_array.shape[0]
    Npix = imgs_array.shape[1]
    Nlc = int(np.sqrt(Nimg))
    Npix_out = Nlc*Npix
    one_big = np.zeros((Npix_out, Npix_out))
    ind = 0
    for xax in range(Nlc):
        for yax in range(Nlc):
            one_big[Npix*xax:Npix*(xax+1), Npix*yax:Npix*(yax+1)] = imgs_array[ind, :, :, 0]
            ind = ind+1
    return one_big

def low_pass_filter(npix_tot, npix1, npix2=None):
    if not npix2:
        npix2 = npix_tot
    pix = np.arange(npix_tot)
    filt = 0.5*(1-np.cos(np.pi*(npix2-pix)/(npix2-npix1)))
    filt[0:npix1] = 1.
    return np.array([filt])

def high_pass_filter(npix_tot, npix2, npix1=None):
    if not npix1:
        npix1 = 0
    pix = np.arange(npix_tot)
    filt = 0.5*(1-np.cos(np.pi*(pix-npix1)/(npix2-npix1)))
    filt[npix2:npix_tot] = 1.
    return np.array([filt])

def apodize_and_unify_images(imgs_oversamp, step=32):
    imgs_array = imgs_oversamp[0]
    weights = imgs_oversamp[1]
    indx = imgs_oversamp[2]
    Npix = imgs_array.shape[1]
    lpf = low_pass_filter(Npix, Npix-step)
    hpf = high_pass_filter(Npix, step)
    Nimg = imgs_array.shape[0]
    Nlc = int(np.sqrt(Nimg))
    temp_image = np.zeros((weights.shape[0], Npix*Nlc))
    for nl in range(Nlc):
        for i in range(Nlc):
            if nl==0:
                temp_image[indx[nl, 0]:indx[nl, 1], Npix*i:(Npix*i)+Npix] += imgs_array[nl*Nlc+i]*lpf.T
            elif nl==Nlc-1:
                temp_image[indx[nl, 0]:indx[nl, 1], Npix*i:(Npix*i)+Npix] += imgs_array[nl*Nlc+i]*hpf.T
            else:
                temp_image[indx[nl, 0]:indx[nl, 1], Npix*i:(Npix*i)+Npix] += imgs_array[nl*Nlc+i]*hpf.T*lpf.T
    out_image = np.zeros((weights.shape[0], weights.shape[1]))
    for nl in range(Nlc):
            if nl==0:
                out_image[:, indx[nl, 0]:indx[nl, 1]] += temp_image[:,Npix*nl:(Npix*nl)+Npix]*lpf
            elif nl==Nlc-1:
                out_image[:, indx[nl, 0]:indx[nl, 1]] += temp_image[:,Npix*nl:(Npix*nl)+Npix]*hpf
            else:
                out_image[:, indx[nl, 0]:indx[nl, 1]] += temp_image[:,Npix*nl:(Npix*nl)+Npix]*hpf*lpf
    return out_image

