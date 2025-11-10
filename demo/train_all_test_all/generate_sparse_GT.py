import os
import sys
import numpy as np
import h5py
from scipy.io import savemat, loadmat
from scipy import sparse
import glob
 
# Ensure repo root (containing 'suns') is on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from suns.config import (
    DATAFOLDER_SETS,
    ACTIVE_EXP_SET,
    EXP_ID_SETS,
    OUTPUT_FOLDER,
    RATE_HZ,
    MAG,
)


if __name__ == '__main__':
    # Set the path of the 'GT Masks' folder, which contains the manual labels in 3D arrays.
    dir_Masks = os.path.join(DATAFOLDER_SETS[ACTIVE_EXP_SET], 'GT Masks')

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
