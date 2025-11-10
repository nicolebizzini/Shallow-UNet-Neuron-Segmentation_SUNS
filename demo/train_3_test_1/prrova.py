from scipy.io import savemat, loadmat
import os

Optimization_Info = loadmat('/gpfs/data/shohamlab/nicole/code/SUNS_nicole/training results/output_masks/Optimization_Info_10.mat')
Params_post_mat = Optimization_Info['Params'][0]
print(Params_post_mat['avgArea'])