#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from scipy.io import loadmat, savemat
import multiprocessing as mp

# Ensure project root is on path for `suns` imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from suns.run_suns import suns_batch
from suns import config as suns_config

import tensorflow as tf


def build_params_pre(rate_hz: float, mag: float) -> dict:
    # Match the pre-processing used in train/test demo scripts
    gauss_filt_size = 50 * mag
    num_median_approx = 1000
    decay = 1.25
    leng_tf = int(np.ceil(rate_hz * decay) + 1)
    Poisson_filt = np.exp(-np.arange(leng_tf) / rate_hz / decay)
    Poisson_filt = (Poisson_filt / Poisson_filt.sum()).astype(np.float32)
    return {
        'gauss_filt_size': gauss_filt_size,
        'num_median_approx': num_median_approx,
        'Poisson_filt': Poisson_filt,
    }


def load_params_post(opt_path: str) -> dict:
    md = loadmat(opt_path)
    P = md['Params'][0]
    return {
        'minArea': float(P['minArea'][0][0, 0]),
        'avgArea': float(P['avgArea'][0][0, 0]),
        'thresh_pmap': float(P['thresh_pmap'][0][0, 0]),
        'thresh_mask': float(P['thresh_mask'][0][0, 0]),
        'thresh_COM0': float(P['thresh_COM0'][0][0, 0]),
        'thresh_COM': float(P['thresh_COM'][0][0, 0]),
        'thresh_IOU': float(P['thresh_IOU'][0][0, 0]),
        'thresh_consume': float(P['thresh_consume'][0][0, 0]),
        'cons': int(P['cons'][0][0, 0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Export Output_Masks_<exp_id>.mat under output_IOU_02/output_masks')
    parser.add_argument('--exp_id', default='mouse7_773', help='Video id to process (default: mouse7_773)')
    parser.add_argument('--cv', type=int, default=0, help='CV index to use for model/params (default: 0)')
    args = parser.parse_args()

    exp_set = suns_config.ACTIVE_EXP_SET
    dir_video = suns_config.DATAFOLDER_SETS[exp_set]
    dir_parent = os.path.join(dir_video, suns_config.OUTPUT_FOLDER[exp_set])
    dir_output = os.path.join(dir_parent, 'output_masks')
    dir_weights = os.path.join(dir_parent, 'Weights')
    os.makedirs(dir_output, exist_ok=True)

    # Resolve model and optimization info paths (CV index default 0)
    model_path = os.path.join(dir_weights, f'Model_CV{args.cv}.h5')
    opt_path = os.path.join(dir_output, f'Optimization_Info_{args.cv}.mat')
    if not os.path.exists(opt_path):
        # fall back to the latest available Optimization_Info_*.mat
        cands = sorted([p for p in os.listdir(dir_output) if p.startswith('Optimization_Info_') and p.endswith('.mat')])
        if cands:
            opt_path = os.path.join(dir_output, cands[-1])

    assert os.path.exists(model_path), f'Missing model: {model_path}'
    assert os.path.exists(opt_path), f'Missing Optimization_Info: {opt_path}'

    rate_hz = suns_config.RATE_HZ[exp_set]
    mag = suns_config.MAG[exp_set]
    Params_pre = build_params_pre(rate_hz, mag)
    Params_post = load_params_post(opt_path)

    # Make TF GPU memory growth friendly
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    print(f'Processing {args.exp_id} with:')
    print(f'  model: {model_path}')
    print(f'  opt:   {opt_path}')
    print(f'  out:   {dir_output}')

    # Use multiprocessing to match complete_segment's default useMP=True path
    with mp.Pool() as p:
        Masks, Masks_2, times_active, _, _ = suns_batch(
            dir_video=dir_video,
            Exp_ID=args.exp_id,
            filename_CNN=model_path,
            Params_pre=Params_pre,
            Params_post=Params_post,
            batch_size_eval=1,
            display=True,
            p=p,
        )

    # Save MATLAB file as (Lx, Ly, n)
    Masks_out = Masks.astype(np.uint8).transpose(1, 2, 0)
    times_active_cell = np.empty((len(times_active),), dtype=object)
    for i, ta in enumerate(times_active):
        times_active_cell[i] = np.asarray(ta, dtype=np.int32)

    out_path = os.path.join(dir_output, f'Output_Masks_{args.exp_id}.mat')
    savemat(out_path, {'Masks': Masks_out, 'times_active': times_active_cell}, do_compression=True)
    print(f'✓ Saved {out_path} | shape: {Masks_out.shape}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


