from forse.networks.dcgan import *
import matplotlib.pyplot as plt
from forse.tools.nn_tools import *
from forse.tools.img_tools import *
from forse.tools.mix_tools import *

dcgan = DCGAN(output_directory='/global/homes/k/krach/scratch/NNforFG/DCGAN/opt/210120_adam09', img_size=(64, 64))
training_path = '/global/homes/k/krach/scratch/NNforFG/training_set/'
training_file = 'training_set_1000patches_4x4deg_T_HR12amin_LR1deg_Npix64_lat30.npy'
patch_file = training_path+training_file
dcgan.train(epochs=100000, patches_file=patch_file, batch_size=32, save_interval=5000)
