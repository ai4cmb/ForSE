import healpy as hp
import pylab as pl
import astropy
from astropy import units as u
import collections
import reproject
import numpy as np
import astropy.io.fits as fits

import argparse
import time
import warnings
warnings.filterwarnings("ignore")
from projection_utils import (
                    get_lonlat, get_lonlat_adaptive,
                     reproject2fullsky,  make_mosaic_from_healpix  )

def main(args):
    Npix= pl.int_(args.npix )
    pixel_size = args.pixelsize  *u.arcmin
    overlap = args.overlap *u.deg
    nside_in=args.nside

    hpxsize  = hp.nside2resol(nside_in, arcmin=True ) *u.arcmin
    nside_out = pl.int_( nside_in )

    if args.flat2hpx :
        """
        I assume that each set of square patches encode just T or Q or U maps,
        if you want a TQU .fits map  hpx reprojected map needs to further postprocessing

        """
        if args.verbose : print(f"reading patches from {args.flat_projection}")
        patches    = pl.load(args.flat_projection , allow_pickle=True)
        size_patch = pixel_size.to(u.deg) *Npix
        if args.adaptive_reprojection :
            lon,lat =get_lonlat_adaptive(size_patch , overlap  )

        else:
            lon,lat =get_lonlat(size_patch , overlap )
        filename =args.flat_projection .replace( '.npy','.fits')
        if args.verbose :
            print("reprojecting back to HPX")
            print (f"files will be stored in {filename}")

        s= time.clock()
        newmap, weightsmap = reproject2fullsky(  tiles=patches, lon=lon, lat=lat,
                                            nside_out=nside_out, pixel_size=pixel_size ,
                                            apodization_file =args.apodization_file  ,
                                             Npix = Npix, verbose=True ,
                                                )
        e= time.clock ()
        if args.apodization_file is not None:
            try :
                apomap= hp.read_map(args.apodization_file .replace('.npy', '.fits'), verbose= args.verbose)
                print('Apodized  map already saved ')

            except FileNotFoundError:

                apomap, _ = reproject2fullsky(  tiles=np.ones_like(patches) , lon=lon, lat=lat,
                                                nside_out=nside_out, pixel_size=pixel_size ,
                                                apodization_file =args.apodization_file  ,
                                                Npix = Npix, verbose=True
                                                     )

                hp.write_map( args.apodization_file .replace('.npy', '.fits') , apomap  , overwrite=True  )
            hp.write_map(filename  , [newmap /apomap  , newmap, weightsmap], overwrite=True  )
        else:
            hp.write_map(filename  , [newmap /weightsmap   , newmap, weightsmap], overwrite=True  )

        if args.verbose : print(f"process took {e-s} sec ")

    elif args.hpx2flat :
        if args.has_polarization :
            inputmap = hp.read_map(args.hpxmap, verbose =args.verbose, field=[0,1,2] )
            stringmap ='TQU'
        else:
            stringmap='T'
            inputmap = [ hp.read_map(args.hpxmap, verbose =args.verbose )    ]

        filename  = args.hpxmap.replace('.fits','.npy')
        assert len(stringmap)== len(inputmap )
        assert  nside_in == hp.get_nside(inputmap)

        if args.verbose :
            print(f"Making square tile patches {pixel_size.to(u.deg) *Npix } x {pixel_size.to(u.deg) *Npix } from {args.hpxmap}")
            print (f"files will be stored in {filename}")
        for   imap,maptype   in zip(inputmap, stringmap ) :

            s= time.clock()
            patches, lon, lat = make_mosaic_from_healpix(  imap, Npix, pixel_size.to(u.deg) , overlap=  overlap ,adaptive=args.adaptive_reprojection   )
            e= time.clock ()

            pl.save(filename.replace('.npy',f'_{maptype}.npy') , [patches, lon , lat  ]  )
            if args.verbose : print(f"process took {e-s} sec ")

    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser(  description="Script to perform the projection in serial. This should take few minutes  to \
                    project a healpix map into flat coordinates (nside=2048), and ~8 hours  to reproject it back into healpix." )
    parser.add_argument("--hpxmap" , help='path to the healpix map to be stacked' )
    parser.add_argument("--pixelsize", help = 'pixel size in arcminutes of the input map', type=np.float , default = 3.75 )
    parser.add_argument("--npix", help='size of patches', default = 320, type = np.int )
    parser.add_argument("--nside", help='nside of output map ', default = 2048, type = np.int )
    parser.add_argument("--overlap", help='partial patch overlap in deg', default=5, type=np.float)
    parser.add_argument("--flat2hpx", action="store_true" , default=False )
    parser.add_argument("--hpx2flat", action="store_true" , default=False )
    parser.add_argument("--verbose", action="store_true" , default=False  )
    parser.add_argument("--flat-projection",  help='path to the file with list of patches  ', default ='' )
    parser.add_argument("--has-polarization",  help='include polarization', default =False, action="store_true"  )
    parser.add_argument("--apodization-file",  help='path of the apodization mask', default =None   )
    args = parser.parse_args()
    main( args)
