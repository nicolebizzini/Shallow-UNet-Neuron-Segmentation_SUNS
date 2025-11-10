#!/usr/bin/env python3
"""
Build ground-truth masks from Suite2p ROIs (Fall_05.mat) and save as
FinalMasks_<exp_id>.mat plus a sparse companion file ..._sparse.mat.

Inputs (hardcoded for mouse7 suite2p GT):
- Suite2p plane dir: /gpfs/data/shohamlab/nicole/tifdata/stacked/main/line3/20191109_mouse7_region1/suite2p/plane0
- Fall_05.mat inside that folder

Outputs:
- /gpfs/home/bizzin01/nicole/code/SUNS_nicole/demo/mouse7_suite2pGT/GT Masks/
    FinalMasks_mouse7_773.mat
    FinalMasks_mouse7_773_sparse.mat
    FinalMasks_mouse7_774.mat
    FinalMasks_mouse7_774_sparse.mat
    FinalMasks_mouse7_775.mat
    FinalMasks_mouse7_775_sparse.mat
    FinalMasks_mouse7_776.mat
    FinalMasks_mouse7_776_sparse.mat

Notes:
- We export dense masks in MATLAB-friendly order (Ly, Lx, n) under key 'FinalMasks'.
- We also export a sparse CSC matrix under key 'GTMasks_2' with shape (Ly*Lx, n_valid).
- If Suite2p provides 'iscell', we include only ROIs with iscell[:, 0] > 0.
"""

import os
import sys
import numpy as np
from scipy.io import loadmat, savemat
from scipy import sparse


INPUT_DIR = "/gpfs/data/shohamlab/nicole/tifdata/stacked/main/line3/20191109_mouse7_region1/suite2p/plane0"
FALL_FILE = os.path.join(INPUT_DIR, "Fall_05.mat")

REPO_ROOT = "/gpfs/home/bizzin01/nicole/code/SUNS_nicole"
OUT_DIR = os.path.join(REPO_ROOT, "demo", "mouse7_suite2pGT", "GT Masks")

# The experiment IDs to emit (use the same masks for each requested ID)
EXP_IDS = [
    "mouse7_773",
    "mouse7_774",
    "mouse7_775",
    "mouse7_776",
]


def _load_fall_mat(fall_mat_path: str):
    """Load the Suite2p Fall.mat file and return (stats_list, Lx, Ly, iscell_mask).

    This loader prioritizes scipy.io.loadmat with simplify_cells when available
    to yield Python-native dict/list structures.
    """
    try:
        md = loadmat(fall_mat_path, simplify_cells=True)
    except TypeError:
        md = loadmat(fall_mat_path, squeeze_me=True, struct_as_record=False)

    if "stat" not in md or "ops" not in md:
        raise KeyError("Fall.mat missing required keys 'stat' and/or 'ops'")

    stats = md["stat"]
    ops = md["ops"]
    iscell = md.get("iscell", None)

    # Normalize ops to dict-like
    if not isinstance(ops, dict):
        try:
            # Suite2p structs can be accessed like attributes
            Lx = int(getattr(ops, "Lx"))
            Ly = int(getattr(ops, "Ly"))
        except Exception as e:
            raise ValueError(f"Could not read ops.Lx/Ly: {e}")
    else:
        Lx = int(ops.get("Lx"))
        Ly = int(ops.get("Ly"))

    # Normalize stats to list of dicts (each has xpix, ypix)
    if isinstance(stats, list):
        stats_list = stats
    else:
        # numpy object array or single struct
        try:
            stats_list = list(stats)
        except Exception:
            stats_list = [stats]

    # iscell: prefer first column if present, else boolean array directly
    iscell_mask = None
    if iscell is not None:
        try:
            arr = np.array(iscell)
            if arr.ndim == 2 and arr.shape[1] >= 1:
                first = arr[:, 0]
            else:
                first = arr
            iscell_mask = np.asarray(first, dtype=bool).reshape(-1)
        except Exception:
            iscell_mask = None

    return stats_list, Lx, Ly, iscell_mask


def _build_masks_from_stats(stats_list, Lx: int, Ly: int, iscell_mask=None) -> np.ndarray:
    """Construct dense masks array shaped (n, Ly, Lx) from Suite2p stats list."""
    n = len(stats_list)
    keep = np.ones(n, dtype=bool)
    if iscell_mask is not None and iscell_mask.shape[0] >= n:
        keep &= iscell_mask[:n]

    # Count valid ROIs to preallocate
    n_valid = int(np.sum(keep))
    masks = np.zeros((n_valid, Ly, Lx), dtype=bool)
    out_idx = 0
    for i, s in enumerate(stats_list):
        if not keep[i]:
            continue
        # Access xpix/ypix whether dict-like or struct-like
        if isinstance(s, dict):
            ypix = np.asarray(s.get("ypix", []), dtype=int)
            xpix = np.asarray(s.get("xpix", []), dtype=int)
        else:
            ypix = np.asarray(getattr(s, "ypix", []), dtype=int)
            xpix = np.asarray(getattr(s, "xpix", []), dtype=int)
        if ypix.size == 0:
            continue
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        if valid.size and not np.all(valid):
            ypix = ypix[valid]
            xpix = xpix[valid]
        masks[out_idx, ypix, xpix] = True
        out_idx += 1

    # If some kept entries were empty, trim trailing zeros
    if out_idx != n_valid:
        masks = masks[:out_idx]
    return masks


def _save_dense_and_sparse(dest_base: str, masks_n_h_w: np.ndarray) -> None:
    """Save dense 'FinalMasks' and sparse 'GTMasks_2' files at dest_base."""
    n, h, w = masks_n_h_w.shape
    dense_hw_n = np.transpose(masks_n_h_w.astype(np.uint8), (1, 2, 0))
    savemat(dest_base + ".mat", {"FinalMasks": dense_hw_n}, do_compression=True)

    flat = masks_n_h_w.reshape(n, h * w)
    keep = np.any(flat, axis=1)
    flat_kept = flat[keep].T  # (h*w, n_valid)
    sparse_mat = sparse.csc_matrix(flat_kept.astype(np.uint8))
    savemat(dest_base + "_sparse.mat", {"GTMasks_2": sparse_mat}, do_compression=True)


def main() -> int:
    if not os.path.isfile(FALL_FILE):
        print(f"! Missing Suite2p Fall file: {FALL_FILE}")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)

    stats_list, Lx, Ly, iscell_mask = _load_fall_mat(FALL_FILE)
    masks = _build_masks_from_stats(stats_list, Lx=Lx, Ly=Ly, iscell_mask=iscell_mask)
    print(f"Loaded ROIs: total={len(stats_list)}, kept={masks.shape[0]}, field=(Ly={Ly}, Lx={Lx})")

    for exp_id in EXP_IDS:
        dest_base = os.path.join(OUT_DIR, f"FinalMasks_{exp_id}")
        _save_dense_and_sparse(dest_base, masks)
        print(f"Saved: {os.path.basename(dest_base)}.mat and _sparse.mat")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


