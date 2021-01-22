from forse.networks.wgan import *
import matplotlib.pyplot as plt
from forse.tools.nn_tools import *
from forse.tools.img_tools import *
from forse.tools.mix_tools import *

wgan = WGAN(output_directory='/global/homes/k/krach/scratch/NNforFG/WGAN/first_tests', img_size=(64, 64))
training_path = '/global/homes/k/krach/scratch/NNforFG/training_set/'
training_file = 'training_set_1000patches_20x20deg_T_HR1deg_LR5deg_Npix64_set2.npy'
patch_file = training_path+training_file
wgan.train(epochs=1000, patches_file=patch_file, batch_size=64, save_interval=10)
