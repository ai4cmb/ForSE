from forse.networks.resunet  import ResUNet
import pylab as pl 
from forse.mmmtools import *
resunet = ResUNet(epochs=1, pretrained=True,  output_directory='/global/cscratch1/sd/giuspugl/workstation/super-res') 
resunet.train(patches_file='/global/cscratch1/sd/giuspugl/workstation/super-res/training_set_1000patches_T_1deg_5deg_Npix64.npy' )