# Notes on ForSE

<u>Work in progress are on the Github ForSe repository in the `workinprog` branch</u>

## Training set: 

##### 2020 March 26 (Nicoletta)

We aim at training the NN with Planck data. For now we are considering Planck dust in temperature, therefore the T map at 353 GHz.
To produce the training set we are considering the regions where the signal to noise of the map is higher than 8 producing a mask as follows:

* Take the full resolution Planck T map at 353 GHz
* Compute the S/N ratio considering the full resolution variance map in T
* Generate a mask that includes all the region with S/N>8
* Smooth the mask with a gaussian beam with FWHM=1° and retain everything is above 0.9
* Combine the obtained mask with an isolatitude mask that cuts out everything at galactic latitude below 10 (to remove the inner part of the Galactic plane)

<img src="mask_T_353.png" alt="img" style="zoom:25%;" />

* The training set is then produced considering the Planck T 353 GHz map at 1° (input) and 12' (output). The training consists in 1000 (998) patches of 64x64 pixels (dimension 4°x4°). A patch is considered if at least 90% if its pixels are in the region available given the above mask

* A very messy notebook that does this is at NERSC in : `/global/homes/k/krach/scratch/NNforFG/ForSE/training_sets/make_training_set.ipynb`

* The file with the patches is in the same folder in: `training_set_998patches_4x4deg_T_HR12amin_LR1deg_Npix64_mask8.npy`  it has shape `(2, 998, 64, 64)` where the first row is the high resolution patch at 12' and the second the low res one at 1°

* Here an example:

  

  <img src="training_exp.png" alt="img" style="zoom:40%;" />

  



## Training DCGAN: 

##### 2020 March 26 (Nicoletta)

First we use the training set generated as described above to train the DCGAN. 

I'm currently trying to update the architecture of the original DCGAN I have been using in the past, as serveral issues where present. In particular, there was a strange warning on the number of trainable parameters, probably due to the way the `trainbale=False` flag was used for the Discriminator when the Generator is trained. 

This update of the DCGAN architecture is currently done one a notebook which is here: `/global/homes/k/krach/scratch/NNforFG/ForSE/DCGAN`in the `build_dcgan.ipynb` file. <!--(ci sono alcune note a riguardo sul mio quaderno in data 26 Febbraio 2020)-->

It seems to work properly if the DCGAN is built in the following way:

```python
    self.discriminator = self.build_discriminator()
    self.discriminator.trainable = True
    self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
    self.generator = self.build_generator()
    self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    z = Input(shape=img_shape)
    img = self.generator(z)
    self.discriminator.trainable = False
    self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
    valid = self.discriminator(img)
    self.combined = Model(z, valid)
    self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    self.discriminator = self.build_discriminator()
    self.discriminator.trainable = True
    self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
```
The GENERATOR, DISCRIMINATOR and COMBINED have the following number of parameters:

<img src="generator.png" style="zoom:30%;" />

<img src="discriminator.png" style="zoom:30%;" />

<img src="combined.png" style="zoom:30%;" />

**<u>Important note:</u>** the above notebook on Jupyter at NERSC seems to work only with kernel `tensorflow-v1.15-0-gpu`!!!

