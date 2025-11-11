# %%
import sys
import os
import random
import time
import glob
import numpy as np
import h5py
from scipy.io import savemat, loadmat
import multiprocessing as mp
import gc

# Set critical environment flags BEFORE importing TensorFlow
os.environ['KERAS_BACKEND'] = 'tensorflow'
# Toggle GPU via env: export SUNS_USE_GPU=1 to enable GPU (default on)
USE_GPU = os.environ.get('SUNS_USE_GPU', '1') == '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ('0' if USE_GPU else '-1')
# Hard-disable XLA to avoid libdevice requirements on clusters
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=off --tf_xla_enable_xla_devices=false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf
try:
    if not USE_GPU:
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        else:
            print('Warning: SUNS_USE_GPU=1 but no GPU visible; using CPU.')
except Exception:
    pass

from suns.Network.data_gen import data_gen
from suns.Network.shallow_unet import get_shallow_unet
from suns.Network.par2 import fastuint
from suns.PostProcessing.complete_post import parameter_optimization


def train_CNN(dir_img, dir_mask, file_CNN, list_Exp_ID_train, list_Exp_ID_val, \
    BATCH_SIZE, NO_OF_EPOCHS, num_train_per, num_total, dims, Params_loss=None, exist_model=None, \
    fine_tune_lr=None, unfreeze_last_k=None, use_early_stopping=True):
    '''Train a CNN model using SNR images in "dir_img" and the corresponding temporal masks in "dir_mask" 
        identified for each video in "list_Exp_ID_train" using tensorflow generater formalism.
        The output are the trained CNN model saved in "file_CNN" and "results" containing loss. 

    Inputs: 
        dir_img (str): The folder containing the network_input (SNR images). 
            Each file must be a ".h5" file, with dataset "network_input" being the SNR video (shape = (T,Lx,Ly)).
        dir_mask (str): The folder containing the temporal masks. 
            Each file must be a ".h5" file, with dataset "temporal_masks" being the temporal masks (shape = (T,Lx,Ly)).
        file_CNN (str): The path to save the trained CNN model.
        list_Exp_ID_train (list of str): The list of file names of the training video(s). 
        list_Exp_ID_val (list of str, default to None): The list of file names of the validation video(s). 
            if list_Exp_ID_val is None, then no validation set is used
        BATCH_SIZE (int): batch size for CNN training.
        NO_OF_EPOCHS (int): number of epochs for CNN training.
        num_train_per (int): number of training images per video.
        num_total (int): total number of frames of a video (can be smaller than acutal number).
        dims (tuplel of int, shape = (2,)): lateral dimension of the video.
        Params_loss(dict, default to None): parameters of the loss function "total_loss"
            Params_loss['DL'](float): Coefficient of dice loss in the total loss
            Params_loss['BCE'](float): Coefficient of binary cross entropy in the total loss
            Params_loss['FL'](float): Coefficient of focal loss in the total loss
            Params_loss['gamma'] (float): first parameter of focal loss
            Params_loss['alpha'] (float): second parameter of focal loss
        exist_model (str, default to None): the path of existing model for transfer learning 
        fine_tune_lr (float, default to None): if set and exist_model provided, recompile with this learning rate
        unfreeze_last_k (int, default to None): if set and exist_model provided, freeze all but last K layers
        use_early_stopping (bool): add EarlyStopping/ReduceLROnPlateau callbacks

    Outputs:
        results: the training results containing the loss information.
        In addition, the trained CNN model is saved in "file_CNN" as ".h5" files.
    '''
    (rows, cols) = dims
    nvideo_train = len(list_Exp_ID_train) # Number of training videos
    # set how to choose training images
    train_every = max(1,num_total//num_train_per)
    start_frame_train = random.randint(0,train_every-1)
    NO_OF_TRAINING_IMAGES = num_train_per * nvideo_train
    
    if list_Exp_ID_val is not None:
        # set how to choose validation images
        nvideo_val = len(list_Exp_ID_val) # Number of validation videos
        # the total number of validation images is about 1/9 of the traning images
        num_val_per = int((num_train_per * nvideo_train / nvideo_val) // 9) 
        num_val_per = min(num_val_per, num_total)
        val_every = num_total//num_val_per
        start_frame_val = random.randint(0,val_every-1)
        NO_OF_VAL_IMAGES = max(num_val_per * nvideo_val, BATCH_SIZE)

    # %% Load traiming images and masks from h5 files
    # training images
    train_imgs = np.zeros((num_train_per * nvideo_train, rows, cols), dtype='float32') 
    # temporal masks for training images
    train_masks = np.zeros((num_train_per * nvideo_train, rows, cols), dtype='uint8') 
    if list_Exp_ID_val is not None:
        # validation images
        val_imgs = np.zeros((num_val_per * nvideo_val, rows, cols), dtype='float32') 
        # temporal masks for validation images
        val_masks = np.zeros((num_val_per * nvideo_val, rows, cols), dtype='uint8') 

    print('Loading training images and masks.')
    # Select training images: for each video, start from frame "start_frame", 
    # select a frame every "train_every" frames, totally "train_val_per" frames  
    for cnt, Exp_ID in enumerate(list_Exp_ID_train):
        h5_img = h5py.File(os.path.join(dir_img, Exp_ID+'.h5'), 'r')
        h5_mask = h5py.File(os.path.join(dir_mask, Exp_ID+'.h5'), 'r')
        num_frame = h5_img['network_input'].shape[0]
        if num_frame >= num_train_per:
            train_imgs[cnt*num_train_per:(cnt+1)*num_train_per,:,:] \
                = np.array(h5_img['network_input'][start_frame_train:train_every*num_train_per:train_every])
            train_masks[cnt*num_train_per:(cnt+1)*num_train_per,:,:] \
                = np.array(h5_mask['temporal_masks'][start_frame_train:train_every*num_train_per:train_every])
        else:
            train_imgs = np.array(h5_img['network_input'])
            train_masks = np.array(h5_mask['temporal_masks'])
        h5_img.close()
        h5_mask.close()

    if list_Exp_ID_val is not None:
        # Select validation images: for each video, start from frame "start_frame", 
        # select a frame every "val_every" frames, totally "num_val_per" frames  
        for cnt, Exp_ID in enumerate(list_Exp_ID_val):
            h5_img = h5py.File(os.path.join(dir_img, Exp_ID+'.h5'), 'r')
            val_imgs[cnt*num_val_per:(cnt+1)*num_val_per,:,:] \
                = np.array(h5_img['network_input'][start_frame_val:val_every*num_val_per:val_every])
            h5_img.close()
            h5_mask = h5py.File(os.path.join(dir_mask, Exp_ID+'.h5'), 'r')
            val_masks[cnt*num_val_per:(cnt+1)*num_val_per,:,:] \
                = np.array(h5_mask['temporal_masks'][start_frame_val:val_every*num_val_per:val_every])
            h5_mask.close()

    # generater for training and validation images and masks
    train_gen = data_gen(train_imgs, train_masks, batch_size=BATCH_SIZE, flips=True, rotate=True)
    if list_Exp_ID_val is not None:
        val_gen = data_gen(val_imgs, val_masks, batch_size=BATCH_SIZE, flips=False, rotate=False)
    
    if list_Exp_ID_val is None:
        val_gen = None
        NO_OF_VAL_IMAGES = 0

    # Ensure XLA JIT is disabled programmatically as well
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    # Disable layout/Remapper optimizers that can force NHWC<->NCHW transposes
    try:
        tf.config.optimizer.set_experimental_options({
            'layout_optimizer': False,
            'remapping': False,
            'constant_folding': True
        })
    except Exception:
        pass
    try:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
        os.environ['TF_USE_LEGACY_KERAS'] = '0'
        os.environ['TF_DISABLE_MKL'] = '1'
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    except Exception:
        pass
    # Also disable XLA experimental runtime if present
    try:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    except Exception:
        pass
    fff = get_shallow_unet(size=None, Params_loss=Params_loss)
    # The alternative line has more options to choose
    # fff = get_shallow_unet_more(size=None, n_depth=3, n_channel=4, skip=[1], activation='elu', Params_loss=Params_loss)
    if exist_model is not None:
        fff.load_weights(exist_model)
        if unfreeze_last_k is not None and unfreeze_last_k > 0:
            cutoff = max(0, len(fff.layers) - int(unfreeze_last_k))
            for li, layer in enumerate(fff.layers):
                layer.trainable = (li >= cutoff)
        if fine_tune_lr is not None:
            # Recompile with a smaller LR for fine-tuning using legacy Adam to avoid XLA-only update paths on some clusters
            try:
                opt_ft = tf.keras.optimizers.legacy.Adam(learning_rate=fine_tune_lr)
            except Exception:
                opt_ft = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
            fff.compile(optimizer=opt_ft, loss=fff.loss, metrics=[m for m in fff.metrics], jit_compile=False)


    class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\n\nThe average loss for epoch {} is {:7.4f}.'.format(epoch, logs['loss']))
    # train CNN
    callbacks = [LossAndErrorPrintingCallback()]
    if use_early_stopping:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss' if val_gen is not None else 'loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1))
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss' if val_gen is not None else 'loss', patience=12, restore_best_weights=True, verbose=1))

    results = fff.fit(train_gen, epochs=NO_OF_EPOCHS, steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                      validation_data=val_gen, validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), verbose=1, callbacks=callbacks)

    # save trained CNN model 
    fff.save_weights(file_CNN)
    return results


def parameter_optimization_pipeline(file_CNN, network_input, dims, \
        Params_set, filename_GT, batch_size_eval=1, useWT=False, useMP=False, p=None):
    '''The complete parameter optimization pipeline for one video and one CNN model.
        It first infers the probablity map of every frame in "network_input" using the trained CNN model in "file_CNN", 
        then calculates the recall, precision, and F1 over all parameter combinations from "Params_set"
        by compairing with the GT labels in "filename_GT". 

    Inputs: 
        file_CNN (str): The path of the trained CNN model. Must be a ".h5" file. 
        network_input (3D numpy.ndarray of float32, shape = (T,Lx,Ly)): 
            the SNR video obtained after pre-processing.
        dims (tuplel of int, shape = (2,)): lateral dimension of the raw video.
        Params_set (dict): Ranges of post-processing parameters to optimize over.
            Params_set['list_minArea']: (list) Range of minimum area of a valid neuron mask (unit: pixels).
            Params_set['list_avgArea']: (list) Range of  typical neuron area (unit: pixels).
            Params_set['list_thresh_pmap']: (list) Range of probablity threshold. 
            Params_set['thresh_mask']: (float) Threashold to binarize the real-number mask.
            Params_set['thresh_COM0']: (float) Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
            Params_set['list_thresh_COM']: (list) Range of threshold of COM distance (unit: pixels) used for the second COM-based merging. 
            Params_set['list_thresh_IOU']: (list) Range of threshold of IOU used for merging neurons.
            Params_set['thresh_consume']: (float) Threshold of consume ratio used for merging neurons.
            Params_set['list_cons']: (list) Range of minimum number of consecutive frames that a neuron should be active for.
        filename_GT (str): file name of the GT masks. 
            The file must be a ".mat" file, with dataset "GTMasks" being the 2D sparse matrix 
            (shape = (Ly0,Lx0,n) when saved in MATLAB).
        batch_size_eval (int, default to 1): batch size of CNN inference.
        useWT (bool, default to False): Indicator of whether watershed is used. 
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        p (multiprocessing.Pool, default to None): 

    Outputs:
        list_Recall (6D numpy.array of float): Recall for all paramter combinations. 
        list_Precision (6D numpy.array of float): Precision for all paramter combinations. 
        list_F1 (6D numpy.array of float): F1 for all paramter combinations. 
            For these outputs, the orders of the tunable parameters are:
            "minArea", "avgArea", "thresh_pmap", "thresh_COM", "thresh_IOU", "cons"
    '''
    (Lx, Ly) = dims
    # load CNN model
    fff = get_shallow_unet()
    fff.load_weights(file_CNN)

    # CNN inference
    start_test = time.time()
    prob_map = fff.predict(network_input, batch_size=batch_size_eval)
    finish_test = time.time()
    Time_frame = (finish_test-start_test)/network_input.shape[0]*1000
    print('Average infrence time {} ms/frame'.format(Time_frame))

# -------- SAVE raw p-map (pre post-processing) into output_masks/<Exp_ID>/ --------
    import os, h5py, numpy as np
    # squeeze to (T, Lx, Ly) and keep float32 precision in [0,1]
    prob_map_f32 = prob_map.squeeze(axis=-1)[:,:Lx,:Ly].astype('float32')

    # Recover Exp_ID from GT filename (FinalMasks_<Exp_ID>[_sparse].mat)
    try:
        base = os.path.basename(filename_GT)
        Exp_ID = base.replace('FinalMasks_','').replace('_sparse','').replace('.mat','')
    except Exception:
        Exp_ID = 'unknown'

    # Extract CV tag from weights filename (Model_CV{n}.h5)
    try:
        cv_tag = os.path.splitext(os.path.basename(file_CNN))[0].replace('Model_','')  # e.g., "CV0"
    except Exception:
        cv_tag = 'CVX'

    # Compute dir_parent from weights path, then use output_masks/<Exp_ID>
    dir_parent = os.path.dirname(os.path.dirname(file_CNN))         # .../<parent>/
    pmap_dir   = os.path.join(dir_parent, 'output_masks', Exp_ID)   # .../output_masks/<Exp_ID>/
    os.makedirs(pmap_dir, exist_ok=True)

    raw_path = os.path.join(pmap_dir, f'{Exp_ID}_{cv_tag}_pmap_raw.h5')
    max_path = os.path.join(pmap_dir, f'{Exp_ID}_{cv_tag}_pmap_max.h5')

    with h5py.File(raw_path, 'w') as f:
        f.create_dataset('pmap', data=prob_map_f32, compression='gzip')     # (T, Lx, Ly) float32
    with h5py.File(max_path, 'w') as f:
        f.create_dataset('pmap_max', data=prob_map_f32.max(axis=0), compression='gzip')  # (Lx, Ly) float32

    print(f'[pmap] saved: {raw_path}')
    print(f'[pmap] saved: {max_path}')
    
    # -------------------end saving pmp-----------------------------

    # convert the output probability map from float to uint8 to speed up future parameter optimization
    prob_map = prob_map.squeeze(axis=-1)[:,:Lx,:Ly]
    pmaps = np.zeros(prob_map.shape, dtype='uint8')
    fastuint(prob_map, pmaps)
    del prob_map, fff

    # calculate the recall, precision, and F1 when different post-processing hyper-parameters are used.
    list_Recall, list_Precision, list_F1 = parameter_optimization(pmaps, Params_set, filename_GT, useMP=useMP, useWT=useWT, p=p)
    return list_Recall, list_Precision, list_F1


def parameter_optimization_cross_validation(cross_validation, list_Exp_ID, Params_set, \
        dims, dir_img, weights_path, dir_GTMasks, dir_temp, dir_output, \
            batch_size_eval=1, useWT=False, useMP=True, load_exist=False, max_eid=None):
    '''The parameter optimization for a complete cross validation.
        For each cross validation, it uses "parameter_optimization_pipeline" to calculate 
        the recall, precision, and F1 of each training video over all parameter combinations from "Params_set",
        and search the parameter combination that yields the highest average F1 over all the training videos. 
        The results are saved in "dir_temp" and "dir_output". 

    Inputs: 
        cross_validation (str, can be "leave-one-out", "train_1_test_rest", or "use_all"): 
            Represent the cross validation type:
                "leave-one-out" means training on all but one video and testing on that one video;
                "train_1_test_rest" means training on one video and testing on the other videos;
                "use_all" means training on all videos and testing on other videos not in the list.
        list_Exp_ID (list of str): The list of file names of all the videos. 
        Params_set (dict): Ranges of post-processing parameters to optimize over.
            Params_set['list_minArea']: (list) Range of minimum area of a valid neuron mask (unit: pixels).
            Params_set['list_avgArea']: (list) Range of  typical neuron area (unit: pixels).
            Params_set['list_thresh_pmap']: (list) Range of probablity threshold. 
            Params_set['thresh_mask']: (float) Threashold to binarize the real-number mask.
            Params_set['thresh_COM0']: (float) Threshold of COM distance (unit: pixels) used for the first COM-based merging. 
            Params_set['list_thresh_COM']: (list) Range of threshold of COM distance (unit: pixels) used for the second COM-based merging. 
            Params_set['list_thresh_IOU']: (list) Range of threshold of IOU used for merging neurons.
            Params_set['thresh_consume']: (float) Threshold of consume ratio used for merging neurons.
            Params_set['list_cons']: (list) Range of minimum number of consecutive frames that a neuron should be active for.
        dims (tuplel of int, shape = (2,)): lateral dimension of the raw video.
        dir_img (str): The path containing the SNR video after pre-processing.
            Each file must be a ".h5" file, with dataset "network_input" being the SNR video (shape = (T,Lx,Ly)).
        weights_path (str): The path containing the trained CNN model, saved as ".h5" files.
        dir_GTMasks (str): The path containing the GT masks.
            Each file must be a ".mat" file, with dataset "GTMasks" being the 2D sparse matrix
            (shape = (Ly0,Lx0,n) when saved in MATLAB).
        dir_temp (str): The path to save the recall, precision, and F1 of various parameters.
        dir_output (str): The path to save the optimal parameters.
        batch_size_eval (int, default to 1): batch size of CNN inference.
        useWT (bool, default to False): Indicator of whether watershed is used. 
        useMP (bool, defaut to True): indicator of whether multiprocessing is used to speed up. 
        load_exist (bool, default to False): Indicator of whether previous F1 of various parameters are loaded. 
        max_eid (int, default to None): The maximum index of video to process. 
            If it is not None, this limits the number of processed video, so that the entire process can be split into multiple scripts. 

    Outputs:
        No output variable, but the recall, precision, and F1 of various parameters 
            are saved in folder "dir_temp" as "Parameter Optimization CV() Exp().mat"
            and the optimal parameters are saved in folder "dir_output" as "Optimization_Info_().mat"
    '''
    nvideo = len(list_Exp_ID) # number of videos used for cross validation
    if cross_validation == "leave_one_out":
        nvideo_train = nvideo-1
    elif cross_validation == "train_1_test_rest":
        nvideo_train = 1
    elif cross_validation == 'use_all':
        nvideo_train = nvideo
    else:
        raise('wrong "cross_validation"')
    (Lx, Ly) = dims

    list_minArea = Params_set['list_minArea']
    list_avgArea = Params_set['list_avgArea']
    list_thresh_pmap = Params_set['list_thresh_pmap']
    thresh_COM0 = Params_set['thresh_COM0']
    list_thresh_COM = Params_set['list_thresh_COM']
    list_thresh_IOU = Params_set['list_thresh_IOU']
    thresh_mask = Params_set['thresh_mask']
    list_cons = Params_set['list_cons']

    if cross_validation == 'use_all':
        size_F1 = (nvideo+1,nvideo,len(list_minArea),len(list_avgArea),len(list_thresh_pmap),len(list_thresh_COM),len(list_thresh_IOU),len(list_cons))
        # arrays to save the recall, precision, and F1 when different post-processing hyper-parameters are used.
    else:
        size_F1 = (nvideo,nvideo,len(list_minArea),len(list_avgArea),len(list_thresh_pmap),len(list_thresh_COM),len(list_thresh_IOU),len(list_cons))

    F1_train = np.zeros(size_F1)
    Recall_train = np.zeros(size_F1)
    Precision_train = np.zeros(size_F1)
    (array_AvgArea, array_minArea, array_thresh_pmap, array_thresh_COM, array_thresh_IOU, array_cons)\
        =np.meshgrid(list_avgArea, list_minArea, list_thresh_pmap, list_thresh_COM, list_thresh_IOU, list_cons)
        # Notice that meshgrid swaps the first two dimensions, so they are placed in a different way.

    # %% start parameter optimization for each video with various CNN models
    p = mp.Pool(mp.cpu_count())
    for (eid,Exp_ID) in enumerate(list_Exp_ID):
        if max_eid is not None:
            if eid > max_eid:
                continue
        gc.collect()
        list_saved_results = glob.glob(os.path.join(dir_temp, 'Parameter Optimization CV* Exp{}.mat'.format(Exp_ID)))
        saved_results_CVall = os.path.join(dir_temp, 'Parameter Optimization CV{} Exp{}.mat'.format(nvideo, Exp_ID))
        if saved_results_CVall in list_saved_results:
            num_exist = len(list_saved_results)-1
        else:
            num_exist = len(list_saved_results)

        if not load_exist or num_exist<nvideo_train: 
            # load SNR videos as "network_input"
            network_input = 0
            print('Video '+Exp_ID)
            start = time.time()
            h5_img = h5py.File(os.path.join(dir_img, Exp_ID+'.h5'), 'r')
            (nframes, rows, cols) = h5_img['network_input'].shape
            network_input = np.zeros((nframes, rows, cols, 1), dtype='float32')
            for t in range(nframes):
                network_input[t, :,:,0] = np.array(h5_img['network_input'][t])
            h5_img.close()
            time_load = time.time()
            filename_GT = dir_GTMasks + Exp_ID + '_sparse.mat'
            print('Load data: {} s'.format(time_load-start))

        if cross_validation == "leave_one_out":
            list_CV = list(range(nvideo))
            list_CV.pop(eid)
        elif cross_validation == "train_1_test_rest":
            list_CV = [eid]
        else: # cross_validation == 'use_all'
            list_CV = [nvideo]

        for CV in list_CV:
            mat_filename = os.path.join(dir_temp, 'Parameter Optimization CV{} Exp{}.mat'.format(CV,Exp_ID))
            if os.path.exists(mat_filename) and load_exist: 
                # if the temporary output file already exists, load it
                mdict = loadmat(mat_filename)
                Recall_train[CV,eid] = np.array(mdict['list_Recall'])
                Precision_train[CV,eid] = np.array(mdict['list_Precision'])
                F1_train[CV,eid] = np.array(mdict['list_F1'])
        
            else: # Calculate recall, precision, and F1 for various parameters
                start = time.time()
                file_CNN = os.path.join(weights_path, 'Model_CV{}.h5'.format(CV))
                list_Recall, list_Precision, list_F1 = parameter_optimization_pipeline(
                    file_CNN, network_input, (Lx,Ly), Params_set, filename_GT, batch_size_eval, useWT=useWT, useMP=useMP, p=p)
                
                Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
                    array_thresh_COM.ravel(), array_thresh_IOU.ravel(), list_Recall.ravel(), list_Precision.ravel(), list_F1.ravel()]).T
                Recall_train[CV,eid] = list_Recall
                Precision_train[CV,eid] = list_Precision
                F1_train[CV,eid] = list_F1
                # save recall, precision, and F1 in a temporary ".mat" file
                mdict={'list_Recall':list_Recall, 'list_Precision':list_Precision, 'list_F1':list_F1, 'Table':Table, 'Params_set':Params_set}
                savemat(mat_filename, mdict) 

    p.close()
            
    # %% Find the optimal postprocessing parameters
    if cross_validation == 'use_all':
        list_CV = [nvideo]
    else:
        list_CV = list(range(nvideo))
    for CV in list_CV:
        # calculate the mean recall, precision, and F1 of all the training videos
        Recall_mean = Recall_train[CV].mean(axis=0)*nvideo/nvideo_train
        Precision_mean = Precision_train[CV].mean(axis=0)*nvideo/nvideo_train
        F1_mean = F1_train[CV].mean(axis=0)*nvideo/nvideo_train
        Table=np.vstack([array_minArea.ravel(), array_AvgArea.ravel(), array_thresh_pmap.ravel(), array_cons.ravel(), 
            array_thresh_COM.ravel(), array_thresh_IOU.ravel(), Recall_mean.ravel(), Precision_mean.ravel(), F1_mean.ravel()]).T
        print('F1_max=', [x.max() for x in F1_train[CV]])

        # find the post-processing hyper-parameters to achieve the highest average F1 over the training videos
        ind = F1_mean.argmax()
        ind = np.unravel_index(ind,F1_mean.shape)
        minArea = list_minArea[ind[0]]
        avgArea = list_avgArea[ind[1]]
        thresh_pmap = list_thresh_pmap[ind[2]]
        thresh_COM = list_thresh_COM[ind[3]]
        thresh_IOU = list_thresh_IOU[ind[4]]
        thresh_consume = (1+thresh_IOU)/2
        cons = list_cons[ind[5]]
        Params={'minArea': minArea, 'avgArea': avgArea, 'thresh_pmap': thresh_pmap, 'thresh_mask': thresh_mask, 
            'thresh_COM0': thresh_COM0, 'thresh_COM': thresh_COM, 'thresh_IOU': thresh_IOU, 'thresh_consume': thresh_consume, 'cons':cons}
        print(Params)
        print('F1_mean=', F1_mean[ind])

        # save the optimal hyper-parameters to a ".mat" file
        Info_dict = {'Params_set':Params_set, 'Params':Params, 'Table': Table, \
            'Recall_train':Recall_train[CV], 'Precision_train':Precision_train[CV], 'F1_train':F1_train[CV]}
        savemat(os.path.join(dir_output, 'Optimization_Info_{}.mat'.format(CV)), Info_dict)

