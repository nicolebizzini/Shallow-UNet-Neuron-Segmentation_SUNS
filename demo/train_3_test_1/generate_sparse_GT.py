import os
import sys
import numpy as np
import h5py
from scipy.io import savemat, loadmat
from scipy import sparse
import glob

# Add the suns directory to the path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'suns'))
from config import DATAFOLDER_SETS, ACTIVE_EXP_SET

if __name__ == '__main__':
    # Set the path of the 'GT Masks' folder, which contains the manual labels in 3D arrays.
    # Use config to get the active dataset path
    data_folder = DATAFOLDER_SETS[ACTIVE_EXP_SET]
    dir_Masks = os.path.join(data_folder, 'GT Masks')

    # %%
    dir_all = glob.glob(os.path.join(dir_Masks,'*FinalMasks*.mat'))
    for path_name in dir_all:
        file_name = os.path.split(path_name)[1]
        if '_sparse' not in file_name:
            print(file_name)
            try: # If file_name is saved in '-v7.3' format
                mat = h5py.File(path_name,'r')
                FinalMasks = np.array(mat['FinalMasks']).astype('bool')
                mat.close()
            except OSError: # If file_name is not saved in '-v7.3' format
                mat = loadmat(path_name)
                FinalMasks = np.array(mat["FinalMasks"]).transpose([2,1,0]).astype('bool')

            (ncells,Ly,Lx) = FinalMasks.shape
            GTMasks_2=sparse.coo_matrix(FinalMasks.reshape(ncells,Lx*Ly).T)
            savemat(os.path.join(path_name[:-4]+'_sparse.mat'), \
                {'GTMasks_2':GTMasks_2}, do_compression=True)
