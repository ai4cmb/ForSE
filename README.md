# ForSE: Foreground Scale Extender

We provide the neural network architecture, weights, data and python notebooks needed to reproduce the results presented in the following paper:

###### *ForSE: a GAN based algorithm for extending CMB foreground models to sub-degree angular scales*, **Krachmalnicoff, N. & Puglisi, G. (2021), submitted to ApJ,  https://arxiv.org/abs/2011.02221**

We take advantage of a generative adversarial network (GAN) to produce realistic and non-Gaussian small scale features on CMB foreground maps.

In our first application we used this approach to generate full sky polarized thermal dust maps at an angular resolution of 12 arc-minutes. 

The maps are publicly available: https://portal.nersc.gov/project/sobs/users/ForSE/fullsky_maps/

## Dependencies  

- Astropy: https://www.astropy.org
- Healpy: https://healpy.readthedocs.io/en/latest/
- Tensorflow: https://www.tensorflow.org
- Keras: https://keras.io
- reproject (only to perform projection from Healpix maps to flat patches and viceversa): https://pypi.org/project/reproject/
- Namaster (only to compute power spectra): https://namaster.readthedocs.io/en/latest/
- numba (only to accelearte the calculation of Minkowski functionals for a given patch): http://numba.pydata.org/

## Install

The code is still under development. To install, use the following command:

```bash
[sudo] python setup.py develop [--user]
```

## Repo organization

* **forse/networks**: our GAN architecture used to generate small scale thermal dust maps. The network is developed in Keras+Tensorflow.
* **forse/tools:** some useful tools to get patches from Healpix maps and reproject back, to generate training sets and to compute Minkowski functionals.
* **forse/scripts:** scripts to train the network (with the approach explained in the ForSE paper) and to reproject full sky patches on the sphere.
* **forse/notebooks:** python notebooks to reproduce the results presented in the ForSE paper. One notebook for total intensity and one for polarization results.

## Data and NN weights

The weights of the trained GANs (for both polarization and total intensitity) are publicly available:

https://portal.nersc.gov/project/sobs/users/ForSE/NN_datautils/weights/

as well as the data used to train and test the network:

https://portal.nersc.gov/project/sobs/users/ForSE/NN_datautils/datasets/

## Citing

If you use this code or any product, please cite:  [![DOI](https://zenodo.org/badge/333457057.svg)](https://zenodo.org/badge/latestdoi/333457057)

and the following paper: 

*ForSE: a GAN based algorithm for extending CMB foreground models to sub-degree angular scales*, Krachmalnicoff, N. & Puglisi, G. (2021), submitted to ApJ

```
@ARTICLE{2021ApJ...911...42K,
       author = {{Krachmalnicoff}, Nicoletta and {Puglisi}, Giuseppe},
        title = "{ForSE: A GAN-based Algorithm for Extending CMB Foreground Models to Subdegree Angular Scales}",
      journal = {\apj},
     keywords = {Cosmic microwave background radiation, Neural networks, Diffuse radiation, 322, 1933, 383, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2021,
        month = apr,
       volume = {911},
       number = {1},
          eid = {42},
        pages = {42},
          doi = {10.3847/1538-4357/abe71c},
archivePrefix = {arXiv},
       eprint = {2011.02221},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021ApJ...911...42K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## License

The library is released under a MIT license. See the file LICENSE for more information.
