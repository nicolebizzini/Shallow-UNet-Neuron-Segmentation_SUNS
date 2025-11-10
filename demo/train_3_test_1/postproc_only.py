import sys
import os
import math
import numpy as np
import h5py

sys.path.insert(1, '../..')  # the path containing "suns" folder
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set which GPU to use. '-1' uses only CPU.

from suns import config
from suns.PreProcessing.preprocessing_functions import find_dataset
from suns.train_CNN_params import parameter_optimization_cross_validation

import tensorflow as tf
tf_version = int(tf.__version__[0])
if tf_version == 1:
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    sess = tf.Session(config=config_tf)
else:  # tf_version == 2:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    # -------- Use existing config to locate data and videos -------- #
    list_Exp_ID = config.EXP_ID_SETS[config.ACTIVE_EXP_SET]
    dir_video = config.DATAFOLDER_SETS[config.ACTIVE_EXP_SET]
    dir_GTMasks = os.path.join(dir_video, 'GT Masks', 'FinalMasks_')

    # Post-processing hyper-parameter ranges (keep same as training script except IOU)
    list_minArea = list(range(30, 85, 5))
    list_avgArea = [177]
    list_thresh_pmap = list(range(130, 235, 10))
    thresh_mask = 0.3
    thresh_COM0 = 2
    list_thresh_COM = list(np.arange(4, 9, 1))
    # CHANGE ONLY THIS: set IOU threshold to 0.2
    list_thresh_IOU = [0.2]
    list_cons = list(range(1, 8, 1))

    # Processing options
    batch_size_eval = 100
    useWT = False
    useMP = False
    cross_validation = "leave_one_out"

    # Input directories (reused from training layout of ACTIVE_EXP_SET)
    dir_parent = os.path.join(dir_video, config.OUTPUT_FOLDER[config.ACTIVE_EXP_SET])
    dir_network_input = os.path.join(dir_parent, 'network_input')
    weights_path = os.path.join(dir_parent, 'Weights')

    # Output directories: write under mouse7_new/output_IOU_02
    output_base = os.path.join(config.DATAFOLDER_SETS['mouse7_new'], 'output_IOU_02')
    dir_output = os.path.join(output_base, 'output_masks')
    dir_temp = os.path.join(output_base, 'temp')
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)

    # Discover dimensions from the raw videos (must match those used in training)
    nvideo = len(list_Exp_ID)
    list_Dimens = np.zeros((nvideo, 3), dtype='uint16')
    for (eid, Exp_ID) in enumerate(list_Exp_ID):
        h5_video = os.path.join(dir_video, Exp_ID + '.h5')
        with h5py.File(h5_video, 'r') as h5_file:
            dset = find_dataset(h5_file)
            list_Dimens[eid] = h5_file[dset].shape

    nframes = np.unique(list_Dimens[:, 0])
    Lx = np.unique(list_Dimens[:, 1])
    Ly = np.unique(list_Dimens[:, 2])
    if len(Lx) * len(Ly) != 1:
        raise ValueError('All training videos must share the same lateral dimensions.')
    rows = Lx[0]
    cols = Ly[0]

    # Adjust hyper-parameter units according to magnification
    Mag = config.MAG[config.ACTIVE_EXP_SET]
    list_minArea = list(np.round(np.array(list_minArea) * Mag ** 2))
    list_avgArea = list(np.round(np.array(list_avgArea) * Mag ** 2))
    thresh_COM0 = thresh_COM0 * Mag
    list_thresh_COM = list(np.array(list_thresh_COM) * Mag)

    Params_set = {
        'list_minArea': list_minArea,
        'list_avgArea': list_avgArea,
        'list_thresh_pmap': list_thresh_pmap,
        'thresh_COM0': thresh_COM0,
        'list_thresh_COM': list_thresh_COM,
        'list_thresh_IOU': list_thresh_IOU,
        'thresh_mask': thresh_mask,
        'list_cons': list_cons,
    }

    # Run only the post-processing parameter optimization
    parameter_optimization_cross_validation(
        cross_validation,
        list_Exp_ID,
        Params_set,
        (rows, cols),
        dir_network_input,
        weights_path,
        dir_GTMasks,
        dir_temp,
        dir_output,
        batch_size_eval,
        useWT=useWT,
        useMP=useMP,
        load_exist=False,
    )


