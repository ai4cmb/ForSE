from forse.networks.dcgan import *
import matplotlib.pyplot as plt
from forse.tools.nn_tools import *
from forse.tools.img_tools import *
from forse.tools.mix_tools import *

dcgan = DCGAN(output_directory='/global/homes/k/krach/scratch/NNforFG/DCGAN/tests/MHD', img_size=(64, 64))
training_path = '/global/homes/k/krach/scratch/NNforFG/training_set/'
training_file = 'training_set_MHDsims_1000patches_20x20deg_T_HR1deg_LR5deg_Npix64.npy'
patch_file = training_path+training_file
dcgan.train(epochs=100000, patches_file=patch_file, batch_size=32, save_interval=5000, swap=10)
