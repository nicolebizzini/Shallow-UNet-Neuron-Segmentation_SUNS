
import math
import numpy as np
import time
from scipy import special
import h5py
try:
    import fissa
except Exception:
    fissa = None
import os
from scipy.io import loadmat
from scipy import ndimage as ndi


def generate_masks(network_input:np.array, file_mask:str, list_thred_ratio:list, dir_save:str, Exp_ID:str):
    '''Generate temporal masks showing active neurons for each SNR frame in "network_input".
        It calculates the traces of each GT neuron in "file_mask", 
        and uses FISSA to decontaminate the traces. 
        Then it normalizes the decontaminated traces to SNR traces. 
        For each "thred_ratio" in "list_thred_ratio", when the SNR is larger than "thred_ratio", 
        the neuron is considered active at this frame.
        For each frame, it addes all the active neurons to generate the binary temporal masks,
        and save the temporal masks in "dir_save". 

    Inputs: 
        network_input (3D numpy.ndarray of float32, shape = (T,Lx,Ly)): the SNR video obtained after pre-processing.
        file_mask (str): The file path to store the GT masks.
            The GT masks are stored in a ".mat" file, and dataset "FinalMasks" is the GT masks
            (shape = (Ly0,Lx0,n) when saved in MATLAB).
        list_thred_ratio (list of float): A list of SNR threshold used to determine when neurons are active.
        dir_save (str): The folder to save the temporal masks of active neurons 
            and the raw and unmixed traces in hard drive.
        Exp_ID (str): The filer name of the SNR video. 

    Outputs:
        No output variable, but the temporal masks is saved in "dir_save" as a "(Exp_ID).h5" file.
            The saved ".h5" file has a dataset "temporal_masks", 
            which stores the temporal masks of active neurons (dtype = 'bool', shape = (T,Lx,Ly))
        In addition, the raw and unmixed traces before and after FISSA are saved in the same folder
            but a different sub-folder in another "(Exp_ID).h5" file. The ".h5" file has two datasets, 
            "raw_traces" and "unmixed_traces" saving the traces before and after FISSA, respectively. 
    '''
    # Prefer curated sparse GT if available (keeps only valid ROIs)
    use_sparse = False
    sparse_file = file_mask.replace('.mat', '_sparse.mat')
    T, rows_pad, cols_pad = network_input.shape
    try:
        if os.path.exists(sparse_file):
            md = loadmat(sparse_file)
            GT = md.get('GTMasks_2', None)
            if GT is not None:
                from scipy.sparse import csc_matrix
                GT = csc_matrix(GT)
                pixels = rows_pad * cols_pad
                # Orient to (pixels, n)
                if GT.shape[0] == pixels and GT.shape[1] != pixels:
                    GTp = GT
                elif GT.shape[1] == pixels and GT.shape[0] != pixels:
                    GTp = GT.transpose()
                else:
                    GTp = GT  # best effort
                n = GTp.shape[1]
                rois = (GTp > 0).astype(bool).toarray().T.reshape(n, rows_pad, cols_pad)
                use_sparse = True
                print(f"[generate_masks] Using sparse GT: {sparse_file} → n={n}")
        if not use_sparse:
            raise FileNotFoundError
    except Exception:
        # Fall back to dense FinalMasks
        try: # If the ".mat" file is saved in '-v7.3' format
            mat = h5py.File(file_mask,'r')
            arr = np.array(mat['FinalMasks'])  # MATLAB HDF5 order: (Ly, Lx, n)
            mat.close()
            if arr.ndim == 3:
                rois = np.transpose(arr, (2, 0, 1)).astype('bool')  # -> (n, Ly, Lx)
            else:
                rois = arr.astype('bool')
        except OSError: # If the ".mat" file is not saved in '-v7.3' format
            mat = loadmat(file_mask)
            arr = np.array(mat["FinalMasks"])  # MATLAB v7 order preserved: (Ly, Lx, n)
            if arr.ndim == 3:
                rois = np.transpose(arr, (2, 0, 1)).astype('bool')  # -> (n, Ly, Lx)
            else:
                rois = arr.astype('bool')
    (nframesf, rowspad, colspad) = network_input.shape
    (ncells, rows, cols) = rois.shape
    # The lateral shape of "network_input" can be larger than that of "rois" due to padding in pre-processing
    # This step crop "network_input" to match the shape of "rois"
    network_input = network_input[:, :rows, :cols]
    # Debug info
    print(f'[generate_masks] network_input: {network_input.shape}  rois: {rois.shape} (ncells={ncells}, rows={rows}, cols={cols})')

    # Validate and standardize inputs for FISSA
    if network_input.ndim != 3:
        raise ValueError(f'network_input must be 3D (T, rows, cols); got shape={network_input.shape}')
    if rois.ndim != 3:
        raise ValueError(f'rois must be 3D (n, rows, cols); got shape={rois.shape}')
    # Coerce dtypes and contiguity
    network_input = np.ascontiguousarray(network_input.astype(np.float32))
    rois = np.ascontiguousarray(rois.astype(bool))
    # Build ROI list and sanity checks
    roi_list = [rois[j] for j in range(ncells)]
    unique_shapes = {r.shape for r in roi_list}
    empty_count = sum(1 for r in roi_list if not r.any())
    print(f'[generate_masks] ROI dtype={rois.dtype}, unique_shapes={unique_shapes}, empty_rois={empty_count}')
    bad_indices = [j for j, r in enumerate(roi_list) if r.ndim != 2 or r.shape != (rows, cols)]
    if bad_indices:
        raise ValueError(f'Found {len(bad_indices)} malformed ROIs (non-2D or inconsistent shape). Example indices: {bad_indices[:5]}')
    # Quick homogeneous stack check to catch inhomogeneous shapes early
    try:
        _ = np.stack(roi_list[:min(len(roi_list), 3)], axis=0)
    except Exception as e:
        raise ValueError(f'ROI list failed homogeneity stack check: {e}')

    # Helper: simple ROI-mean traces if FISSA is not available or ROI shape unsupported
    def _fallback_simple_traces(net_in: np.ndarray, roi_masks: np.ndarray):
        n_rois = roi_masks.shape[0]
        T = net_in.shape[0]
        traces = np.zeros((n_rois, T), dtype=np.float32)
        frames_2d = net_in[:, :rows, :cols].reshape(T, rows*cols)
        for j in range(n_rois):
            m = roi_masks[j]
            if not m.any():
                continue
            idx = np.flatnonzero(m.reshape(-1))
            traces[j] = frames_2d[:, idx].mean(axis=1).astype(np.float32)
        return traces.copy(), traces.copy()

    # Try FISSA first; if unavailable or ROI format rejected, fall back to simple traces
    try:
        if fissa is None:
            raise RuntimeError('FISSA unavailable')
        folder_FISSA = os.path.join(dir_save, 'FISSA')    
        start = time.time()
        # FISSA expects, per trial, a list of ROIs where each ROI is a list of 2D masks (compartments).
        # We provide a single compartment per ROI: [mask]
        try:
            print(f"[generate_masks] FISSA version: {getattr(fissa, '__version__', 'unknown')}")
        except Exception:
            pass
        roi_groups = [[m] for m in roi_list]
        print(f'[generate_masks] Trying FISSA with roi_list of length {len(roi_list)}; first ROI shape={roi_list[0].shape if len(roi_list)>0 else None}; nested per-ROI compartments={len(roi_groups[0]) if len(roi_groups)>0 else 0}')
        experiment = fissa.Experiment([network_input], [roi_groups], folder_FISSA, ncores_preparation=1)
        experiment.separation_prep(redo=True)
        prep = time.time()
        experiment.separate(redo_prep=False, redo_sep=True)
        finish = time.time()
        experiment.save_to_matlab()
        print('FISSA time: {} s'.format(finish-start))
        print('    Preparation time: {} s'.format(prep-start))
        print('    Separation time: {} s'.format(finish-prep))
        # Extract traces
        raw_traces = np.vstack([experiment.raw[x][0][0] for x in range(ncells)])
        unmixed_traces = np.vstack([experiment.result[x][0][0] for x in range(ncells)])
        del experiment
    except Exception as e:
        # Try a second attempt with transposed data order in case FISSA expects (rows, cols, T)
        try:
            print(f'[generate_masks] First FISSA attempt failed: {e}. Retrying with data transposed to (rows, cols, T).')
            start = time.time()
            data_alt = network_input.transpose(1, 2, 0).copy()
            experiment = fissa.Experiment([data_alt], [roi_groups], folder_FISSA, ncores_preparation=1)
            experiment.separation_prep(redo=True)
            prep = time.time()
            experiment.separate(redo_prep=False, redo_sep=True)
            finish = time.time()
            experiment.save_to_matlab()
            print('FISSA (alt order) time: {} s'.format(finish-start))
            print('    Preparation time: {} s'.format(prep-start))
            print('    Separation time: {} s'.format(finish-prep))
            raw_traces = np.vstack([experiment.raw[x][0][0] for x in range(ncells)])
            unmixed_traces = np.vstack([experiment.result[x][0][0] for x in range(ncells)])
            del experiment
        except Exception as e2:
            print(f'FISSA unavailable or ROI format unsupported after retry ({e2}); falling back to simple ROI traces.')
            raw_traces, unmixed_traces = _fallback_simple_traces(network_input, rois)

    # Save the raw and unmixed traces into a ".h5" file under folder "dir_trace".
    dir_trace = os.path.join(dir_save, "traces")
    if not os.path.exists(dir_trace):
        os.makedirs(dir_trace)        
    f = h5py.File(os.path.join(dir_trace, Exp_ID+".h5"), "w")
    f.create_dataset("raw_traces", data = raw_traces)
    f.create_dataset("unmixed_traces", data = unmixed_traces)
    f.close()

    # Calculate median and median-based std to normalize each trace into SNR trace
    # The median-based std is from the raw trace, because FISSA unmixing can change the noise property.
    med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=1)
    med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=1)
    mu_unmix = med_unmix[0]
    sigma_raw = (med_raw[0]-med_raw[1])/(math.sqrt(2)*special.erfinv(0.5))

    # %% Threshold the SNR trace by each number in "list_thred_ratio" to produce temporal masks
    for thred_ratio in list_thred_ratio:
        start_mask = time.time()
        # Threshold the SNR traces by each number in "list_thred_ratio"
        thred = np.expand_dims(mu_unmix + thred_ratio*sigma_raw, axis=1)
        active = (unmixed_traces > thred).astype('bool')

        # %% Generate temporal masks by summing the binary masks of active neurons
        # The shape of "temporal_masks" matches "network_input", and can be larger than "rois"
        temporal_masks = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
        for t in range(nframesf):
            active_neurons = active[:,t]
            temporal_masks[t, :rows, :cols] = rois[active_neurons,:,:].sum(axis=0)>0
        end_mask = time.time()
        print('Mask creation: {} s'.format(end_mask-start_mask))

        # Save temporal masks in "dir_save" in a ".h5" file
        dir_temporal_masks = os.path.join(dir_save, "temporal_masks({})".format(thred_ratio))
        if not os.path.exists(dir_temporal_masks):
            os.makedirs(dir_temporal_masks) 
        f = h5py.File(os.path.join(dir_temporal_masks, Exp_ID+".h5"), "w")
        f.create_dataset("temporal_masks", data = temporal_masks)
        f.close()
        end_saving = time.time()
        print('Mask saving: {} s'.format(end_saving-end_mask))


def generate_masks_from_traces(file_mask:str, list_thred_ratio:list, dir_save:str, Exp_ID:str):
    '''Generate temporal masks showing active neurons for each SNR frame in "network_input".
        Similar to "generate_masks", but this version uses the traces saved in folder "traces", 
        a previous output of "generate_masks", so it does not redo FISSA and does not need input video.

    Inputs: 
        file_mask (str): The file path to store the GT masks.
            The GT masks are stored in a ".mat" file, and dataset "FinalMasks" is the GT masks
            (shape = (Ly0,Lx0,n) when saved in MATLAB).
        list_thred_ratio (list): A list of SNR threshold used to determine when neurons are active.
        dir_save (str): The folder to save the temporal masks of active neurons 
            and the raw and unmixed traces in hard drive.
        Exp_ID (str): The filer name of the SNR video. 

    Outputs:
        No output variable, but the temporal masks is saved in "dir_save" as a "(Exp_ID).h5" file.
            The saved ".h5" file has a dataset "temporal_masks", 
            which stores the temporal masks of active neurons (dtype = 'bool', shape = (T,Lx,Ly))
    '''
    try: # If the ".mat" file is saved in '-v7.3' format
        mat = h5py.File(file_mask,'r')
        rois = np.array(mat['FinalMasks']).astype('bool')
        mat.close()
    except OSError: # If the ".mat" file is not saved in '-v7.3' format
        mat = loadmat(file_mask)
        rois = np.array(mat["FinalMasks"]).transpose([2,1,0])
    (_, rows, cols) = rois.shape
    rowspad = math.ceil(rows/8)*8  # size of the network input and output
    colspad = math.ceil(cols/8)*8

    # %% Extract raw and unmixed traces from the saved ".h5" file
    dir_trace = os.path.join(dir_save, "traces")
    f = h5py.File(os.path.join(dir_trace, Exp_ID+".h5"), "r")
    raw_traces = np.array(f["raw_traces"])
    unmixed_traces = np.array(f["unmixed_traces"])
    f.close()
    nframesf = unmixed_traces.shape[1]
    
    # Calculate median and median-based std to normalize each trace into SNR trace
    # The median-based std is from the raw trace, because FISSA unmixing can change the noise property.
    med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=1)
    med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=1)
    mu_unmix = med_unmix[0]
    sigma_raw = (med_raw[0]-med_raw[1])/(math.sqrt(2)*special.erfinv(0.5))

    # %% Threshold the SNR trace by each number in "list_thred_ratio" to produce temporal masks
    for thred_ratio in list_thred_ratio:
        start_mask = time.time()
        # Threshold the SNR traces by each number in "list_thred_ratio"
        thred = np.expand_dims(mu_unmix + thred_ratio*sigma_raw, axis=1)
        active = (unmixed_traces > thred).astype('bool')

        # %% Generate temporal masks by summing the binary masks of active neurons
        # The shape of "temporal_masks" matches "network_input", and can be larger than "rois"
        temporal_masks = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
        for t in range(nframesf):
            active_neurons = active[:,t]
            temporal_masks[t, :rows, :cols] = rois[active_neurons,:,:].sum(axis=0)>0
        end_mask = time.time()
        print('Mask creation: {} s'.format(end_mask-start_mask))

        # Save temporal masks in "dir_save" in a ".h5" file
        dir_temporal_masks = os.path.join(dir_save, "temporal_masks({})".format(thred_ratio))
        if not os.path.exists(dir_temporal_masks):
            os.makedirs(dir_temporal_masks) 
        f = h5py.File(os.path.join(dir_temporal_masks, Exp_ID+".h5"), "w")
        f.create_dataset("temporal_masks", data = temporal_masks)
        f.close()
        end_saving = time.time()
        print('Mask saving: {} s'.format(end_saving-end_mask))





# FISSA 
# import math
# import numpy as np
# import time
# from scipy import special
# import h5py
# import fissa
# import os
# from scipy.io import loadmat


# def generate_masks(network_input:np.array, file_mask:str, list_thred_ratio:list, dir_save:str, Exp_ID:str):
#     '''Generate temporal masks showing active neurons for each SNR frame in "network_input".
#         It calculates the traces of each GT neuron in "file_mask", 
#         and uses FISSA to decontaminate the traces. 
#         Then it normalizes the decontaminated traces to SNR traces. 
#         For each "thred_ratio" in "list_thred_ratio", when the SNR is larger than "thred_ratio", 
#         the neuron is considered active at this frame.
#         For each frame, it addes all the active neurons to generate the binary temporal masks,
#         and save the temporal masks in "dir_save". 

#     Inputs: 
#         network_input (3D numpy.ndarray of float32, shape = (T,Lx,Ly)): the SNR video obtained after pre-processing.
#         file_mask (str): The file path to store the GT masks.
#             The GT masks are stored in a ".mat" file, and dataset "FinalMasks" is the GT masks
#             (shape = (Ly0,Lx0,n) when saved in MATLAB).
#         list_thred_ratio (list of float): A list of SNR threshold used to determine when neurons are active.
#         dir_save (str): The folder to save the temporal masks of active neurons 
#             and the raw and unmixed traces in hard drive.
#         Exp_ID (str): The filer name of the SNR video. 

#     Outputs:
#         No output variable, but the temporal masks is saved in "dir_save" as a "(Exp_ID).h5" file.
#             The saved ".h5" file has a dataset "temporal_masks", 
#             which stores the temporal masks of active neurons (dtype = 'bool', shape = (T,Lx,Ly))
#         In addition, the raw and unmixed traces before and after FISSA are saved in the same folder
#             but a different sub-folder in another "(Exp_ID).h5" file. The ".h5" file has two datasets, 
#             "raw_traces" and "unmixed_traces" saving the traces before and after FISSA, respectively. 
#     '''
#   # 1) Load and normalize ROI masks to (n, rows, cols) boolean
#     # -----------------------
#     try:
#         # v7.3 HDF5: typically (rows, cols, n)
#         mat = h5py.File(file_mask, 'r')
#         rois = np.array(mat['FinalMasks'])
#         mat.close()
#         if rois.ndim != 3:
#             raise ValueError("FinalMasks must be 3D")
#         # Standardize to (n, rows, cols)
#         rois = np.transpose(rois, (2, 0, 1)).astype(bool)
#     except OSError:
#         # non-v7.3 MAT
#         mat = loadmat(file_mask)
#         rois = np.array(mat["FinalMasks"])
#         if rois.ndim != 3:
#             raise ValueError("FinalMasks must be 3D")
#         # Many non-v7.3 exports are already (n, rows, cols)
#         # If yours are (rows, cols, n), transpose as needed:
#         # Heuristic: if first dim looks like image height/width (e.g., <= 1024)
#         # and last dim is large (e.g., hundreds of ROIs), treat as (rows, cols, n).
#         if rois.shape[0] <= 2048 and rois.shape[2] > rois.shape[0]:
#             rois = np.transpose(rois, (2, 0, 1))
#         rois = rois.astype(bool)

#     # Shapes
#     (T, Lx_pad, Ly_pad) = network_input.shape
#     (ncells, rows, cols) = rois.shape

#     # Crop video to ROI FOV (network_input may be padded)
#     network_input = network_input[:, :rows, :cols]

#     # -----------------------
#     # 2) Temporary np.save shim for FISSA ragged tuple save
#     # -----------------------
#     import numpy as _np
#     __orig_save = _np.save

#     def _safe_save(fname, arr, *args, **kwargs):
#         try:
#             return __orig_save(fname, arr, *args, **kwargs)
#         except ValueError:
#             # Coerce ragged tuple/list to an object array so np.save can handle it
#             return __orig_save(fname, _np.array(arr, dtype=object), *args, **kwargs)

#     _np.save = _safe_save

#     # -----------------------
#     # 3) FISSA: decontaminate traces
#     # -----------------------
#     folder_FISSA = os.path.join(dir_save, 'FISSA')
#     os.makedirs(folder_FISSA, exist_ok=True)

#     start = time.time()
#     # FISSA expects a list of 2D masks (rows, cols) for each ROI
#     roi_list = list(rois)

#     experiment = fissa.Experiment(
#         [network_input],          # list of trials (each trial: array T x rows x cols)
#         [roi_list],               # list of ROI-lists (one list per trial)
#         folder_FISSA,
#         ncores_preparation=1
#     )
#     # 0.7.2 runs separation_prep() in __init__, so no need to call it again.
#     prep = time.time()

#     experiment.separate(redo_prep=False, redo_sep=True)
#     finish = time.time()
#     #experiment.save_to_matlab()

#     # Optional: restore np.save (keep the shim if you chain multiple runs)
#     _np.save = __orig_save

#     del network_input
#     print('FISSA time: {} s'.format(finish - start))
#     print('    Preparation time: {} s'.format(prep - start))
#     print('    Separation time: {} s'.format(finish - prep))

#     # -----------------------
#     # 4) Extract traces from FISSA output
#     # -----------------------
#     raw_traces = np.vstack([experiment.raw[x][0][0] for x in range(ncells)])
#     unmixed_traces = np.vstack([experiment.result[x][0][0] for x in range(ncells)])
#     del experiment

#     # Save traces
#     dir_trace = os.path.join(dir_save, "traces")
#     os.makedirs(dir_trace, exist_ok=True)
#     with h5py.File(os.path.join(dir_trace, Exp_ID + ".h5"), "w") as f:
#         f.create_dataset("raw_traces", data=raw_traces)
#         f.create_dataset("unmixed_traces", data=unmixed_traces)

#     # -----------------------
#     # 5) Convert to SNR and build temporal masks at each threshold
#     # -----------------------
#     # Median-based noise estimate from raw traces
#     med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=1)
#     med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=1)
#     mu_unmix = med_unmix[0]
#     sigma_raw = (med_raw[0] - med_raw[1]) / (math.sqrt(2) * special.erfinv(0.5))

#     # For each threshold, create binary temporal masks (T, Lx, Ly) where active neurons are OR'ed
#     for thred_ratio in list_thred_ratio:
#         start_mask = time.time()

#         thred = np.expand_dims(mu_unmix + thred_ratio * sigma_raw, axis=1)  # (n,1)
#         active = (unmixed_traces > thred).astype(bool)                       # (n, T)

#         temporal_masks = np.zeros((T, rows, cols), dtype=np.uint8)
#         for t in range(T):
#             active_neurons = active[:, t]
#             if np.any(active_neurons):
#                 temporal_masks[t, :, :] = (rois[active_neurons, :, :].sum(axis=0) > 0)

#         end_mask = time.time()
#         print('Mask creation: {} s'.format(end_mask - start_mask))

#         # Save temporal masks
#         dir_temporal_masks = os.path.join(dir_save, "temporal_masks({})".format(thred_ratio))
#         os.makedirs(dir_temporal_masks, exist_ok=True)
#         with h5py.File(os.path.join(dir_temporal_masks, Exp_ID + ".h5"), "w") as f:
#             f.create_dataset("temporal_masks", data=temporal_masks.astype(bool))

#         end_saving = time.time()
#         print('Mask saving: {} s'.format(end_saving - end_mask))

# def generate_masks_from_traces(file_mask:str, list_thred_ratio:list, dir_save:str, Exp_ID:str):
#     '''Generate temporal masks showing active neurons for each SNR frame in "network_input".
#         Similar to "generate_masks", but this version uses the traces saved in folder "traces", 
#         a previous output of "generate_masks", so it does not redo FISSA and does not need input video.

#     Inputs: 
#         file_mask (str): The file path to store the GT masks.
#             The GT masks are stored in a ".mat" file, and dataset "FinalMasks" is the GT masks
#             (shape = (Ly0,Lx0,n) when saved in MATLAB).
#         list_thred_ratio (list): A list of SNR threshold used to determine when neurons are active.
#         dir_save (str): The folder to save the temporal masks of active neurons 
#             and the raw and unmixed traces in hard drive.
#         Exp_ID (str): The filer name of the SNR video. 

#     Outputs:
#         No output variable, but the temporal masks is saved in "dir_save" as a "(Exp_ID).h5" file.
#             The saved ".h5" file has a dataset "temporal_masks", 
#             which stores the temporal masks of active neurons (dtype = 'bool', shape = (T,Lx,Ly))
#     '''
#     try: # If the ".mat" file is saved in '-v7.3' format
#         mat = h5py.File(file_mask,'r')
#         rois = np.array(mat['FinalMasks']).astype('bool')
#         mat.close()
#     except OSError: # If the ".mat" file is not saved in '-v7.3' format
#         mat = loadmat(file_mask)
#         rois = np.array(mat["FinalMasks"]).transpose([2,1,0])
#     (_, rows, cols) = rois.shape
#     rowspad = math.ceil(rows/8)*8  # size of the network input and output
#     colspad = math.ceil(cols/8)*8

#     # %% Extract raw and unmixed traces from the saved ".h5" file
#     dir_trace = os.path.join(dir_save, "traces")
#     f = h5py.File(os.path.join(dir_trace, Exp_ID+".h5"), "r")
#     raw_traces = np.array(f["raw_traces"])
#     unmixed_traces = np.array(f["unmixed_traces"])
#     f.close()
#     nframesf = unmixed_traces.shape[1]
    
#     # Calculate median and median-based std to normalize each trace into SNR trace
#     # The median-based std is from the raw trace, because FISSA unmixing can change the noise property.
#     med_raw = np.quantile(raw_traces, np.array([0.5, 0.25]), axis=1)
#     med_unmix = np.quantile(unmixed_traces, np.array([0.5, 0.25]), axis=1)
#     mu_unmix = med_unmix[0]
#     sigma_raw = (med_raw[0]-med_raw[1])/(math.sqrt(2)*special.erfinv(0.5))

#     # %% Threshold the SNR trace by each number in "list_thred_ratio" to produce temporal masks
#     for thred_ratio in list_thred_ratio:
#         start_mask = time.time()
#         # Threshold the SNR traces by each number in "list_thred_ratio"
#         thred = np.expand_dims(mu_unmix + thred_ratio*sigma_raw, axis=1)
#         active = (unmixed_traces > thred).astype('bool')

#         # %% Generate temporal masks by summing the binary masks of active neurons
#         # The shape of "temporal_masks" matches "network_input", and can be larger than "rois"
#         temporal_masks = np.zeros((nframesf, rowspad, colspad), dtype='uint8')
#         for t in range(nframesf):
#             active_neurons = active[:,t]
#             temporal_masks[t, :rows, :cols] = rois[active_neurons,:,:].sum(axis=0)>0
#         end_mask = time.time()
#         print('Mask creation: {} s'.format(end_mask-start_mask))

#         # Save temporal masks in "dir_save" in a ".h5" file
#         dir_temporal_masks = os.path.join(dir_save, "temporal_masks({})".format(thred_ratio))
#         if not os.path.exists(dir_temporal_masks):
#             os.makedirs(dir_temporal_masks) 
#         f = h5py.File(os.path.join(dir_temporal_masks, Exp_ID+".h5"), "w")
#         f.create_dataset("temporal_masks", data = temporal_masks)
#         f.close()
#         end_saving = time.time()
#         print('Mask saving: {} s'.format(end_saving-end_mask))



