# Notes on ForSE

### 2020 March 26

**Training set:** 

We aim at training the NN with Planck data. For now we are considering Planck dust in temperature, therefore the T map at 353 GHz.
To produce the training set we are considering the regions where the signal to noise of the map is higher than 8 producing a mask as follows:

* That the full resolution Planck T map at 353 GHz
* compute the S/N ratio considering the full resolution variance map in T
* generate a mask that includes all the region with S/N>8
* smooth the mask with a gaussian beam with FWHM=1° and retain everything is above 0.9
* Combine the obtained mask with an isolatitude mask that cuts out everything at galactic latitude below 10 (to remove the inner part of the Galactic plane)

<img src="mask_T_353.png" alt="img" style="zoom:25%;" />

* The training set is then produced considering the Planck T 353 GHz map at 1° (input) and 12' (output). The training consist in 1000 patches of 64x64 pixels (dimension 4°x4°). A patch is considered if at least 90% if the pixels are in the region available given the above mask

* A very messy notebook that does this is at NERSC in : `/global/homes/k/krach/scratch/NNforFG/ForSE/training_sets/make_training_set.ipynb`

  