from forse.networks.dcgan import *
import matplotlib.pyplot as plt
from forse.tools.nn_tools import *
from forse.tools.img_tools import *
from forse.tools.mix_tools import *
import tensorflow as tf

dcgan = DCGAN(output_directory='./', img_size=(320, 320))
training_path = '/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/'
training_file = 'GNILC_Thr12_Tlr80_350patches_20x20deg_Npix320_mask8.npy'
patch_file = training_path+training_file
dcgan.train(epochs=100000, patches_file=patch_file, batch_size=16, save_interval=1000)

