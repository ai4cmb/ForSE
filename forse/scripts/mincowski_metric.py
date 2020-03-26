import matplotlib.pyplot as plt
from forse.tools.nn_tools import *
from forse.tools.img_tools import *
from forse.tools.mix_tools import *
import healpy as hp
from forse.networks.dcgan import *
from mypy import *

pysm_ss, pysm_1d = np.load(
    patch_path+
    'training_set_PySM_100patches_20x20deg_T_HR12amin_LR1deg_Npix320_lat30.npy')
hfi_ss, hfi_1d = np.load(
    patch_path+
    'training_set_100patches_20x20deg_T_HR12amin_LR1deg_Npix320_lat30.npy')
dir_models = '/global/homes/k/krach/scratch/NNforFG/DCGAN/opt/160120_swap_Dv2/models/'
file_name = 'minkowski_intersection_160120_swap_Dv2.npz'


def compute_mikowski_intersection():
    large_scale_tot = []
    small_scale_tot = []
    NN_small_scale_tot = []
    rhos_nn_ss, f_nn_ss, u_nn_ss, chi_nn_ss = [], [], [], [] 
    rhos_pysm_ss, f_pysm_ss, u_pysm_ss, chi_pysm_ss = [], [], [], [] 
    rhos_hfi_ss, f_hfi_ss, u_hfi_ss, chi_hfi_ss = [], [], [], []
    for s in range(100):
        large_scale = pysm_1d[int(s)]
        small_scale = pysm_ss[int(s)]
        hfi_small_scale = hfi_ss[int(s)]
        hfi_large_scale = hfi_1d[int(s)]
        images_oversamp = divide_image(large_scale, 64, 64)
        images_oversamp_ss = divide_image(small_scale, 64, 64)
        images_oversamp_hfi = divide_image(hfi_large_scale, 64, 64)
        images_oversamp_hfi_ss = divide_image(hfi_small_scale, 64, 64)
        Xos = np.array(images_oversamp[0])
        indx = images_oversamp[2]
        weights = images_oversamp[1]
        for i in range(Xos.shape[0]):
            ratio_oversamp_ss = np.array(images_oversamp_ss[0][i])/np.array(images_oversamp[0][i])
            ratio_oversamp_ss = rescale_min_max(ratio_oversamp_ss)
            ratio_oversamp_hfi_ss = np.array(images_oversamp_hfi_ss[0][i])/np.array(images_oversamp_hfi[0][i])
            ratio_oversamp_hfi_ss = rescale_min_max(ratio_oversamp_hfi_ss)
            rhos_pysm_ss, f_PYSM, u_PYSM, chi_PYSM= get_functionals(ratio_oversamp_ss)
            rhos_hfi_ss, f_hfi, u_hfi, chi_hfi= get_functionals(ratio_oversamp_hfi_ss)
            f_pysm_ss.append(f_PYSM)
            u_pysm_ss.append(u_PYSM)
            chi_pysm_ss.append(chi_PYSM)
            f_hfi_ss.append(f_hfi)
            u_hfi_ss.append(u_hfi)
            chi_hfi_ss.append(chi_hfi)
            Xos[i] = rescale_min_max(Xos[i])
        Xos = Xos.reshape(Xos.shape[0], 64, 64, 1)
        gen_imgs_os = dcgan.generator.predict(Xos)
        nn_images_oversamp = np.copy(images_oversamp)
        nn_images_oversamp[0] = gen_imgs_os[:, :, :, 0]
        for i in range(len(nn_images_oversamp[0])):
            nn_images_oversamp_scal = rescale_min_max(nn_images_oversamp[0][i])
            rhos_nn_ss, f_NN, u_NN, chi_NN= get_functionals(nn_images_oversamp_scal)
            f_nn_ss.append(f_NN)
            u_nn_ss.append(u_NN)
            chi_nn_ss.append(chi_NN)
    f_nn_ss = np.array(f_nn_ss)
    u_nn_ss = np.array(u_nn_ss)
    chi_nn_ss = np.array(chi_nn_ss)
    f_pysm_ss = np.array(f_pysm_ss)
    u_pysm_ss = np.array(u_pysm_ss)
    chi_pysm_ss = np.array(chi_pysm_ss)
    f_hfi_ss = np.array(f_hfi_ss)
    u_hfi_ss = np.array(u_hfi_ss)
    chi_hfi_ss = np.array(chi_hfi_ss)
    m1 = compute_intersection(rhos_hfi_ss, 
                     [np.mean(f_hfi_ss, axis=0)-np.std(f_hfi_ss, axis=0),
                      np.mean(f_hfi_ss, axis=0)+np.std(f_hfi_ss, axis=0)], 
                     [np.mean(f_nn_ss, axis=0)-np.std(f_nn_ss, axis=0),
                      np.mean(f_nn_ss, axis=0)+np.std(f_nn_ss, axis=0)], 
                     npt=100000)
    m2 = compute_intersection(rhos_hfi_ss, 
                     [np.mean(u_hfi_ss, axis=0)-np.std(u_hfi_ss, axis=0), 
                      np.mean(u_hfi_ss, axis=0)+np.std(u_hfi_ss, axis=0)], 
                     [np.mean(u_nn_ss, axis=0)-np.std(u_nn_ss, axis=0),
                      np.mean(u_nn_ss, axis=0)+np.std(u_nn_ss, axis=0)], 
                     npt=100000)
    m3 = compute_intersection(rhos_hfi_ss, 
                     [np.mean(chi_hfi_ss, axis=0)-np.std(chi_hfi_ss, axis=0),
                      np.mean(chi_hfi_ss, axis=0)+np.std(chi_hfi_ss, axis=0)], 
                     [np.mean(chi_nn_ss, axis=0)-np.std(chi_nn_ss,axis=0),
                      np.mean(chi_nn_ss, axis=0)+np.std(chi_nn_ss, axis=0)], 
                     npt=100000)
    return m1, m2, m3

intm1 = []
intm2 = []
intm3 = []
epochs = np.arange(0, 100000, 5000)
for e in epochs:
    dcgan.generator = load_model(dir_models+'generat_'+str(e)+'.h5')
    m1, m2, m3 = compute_mikowski_intersection()
    intm1.append(m1)
    intm2.append(m2)
    intm3.append(m3)
np.savez(
    file_name, 
    epochs = epochs
    int_M1=np.array(intm1), 
    int_M2=np.array(intm2), 
    int_M3=np.array(intm3))
    




    
