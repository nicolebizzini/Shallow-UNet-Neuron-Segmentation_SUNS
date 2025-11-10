#!/usr/bin/env python3
"""
Convert .tif files to .h5 format for line3_dataset (single file per mouse)
This script uses only the first .tif file from each mouse to get (1600, 256, 256) dimensions
"""

import tifffile
import h5py
import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import os
import glob
import re
import sys

# Add the suns directory to the path
sys.path.insert(0, 'suns')
from config import DATAFOLDER_SETS, ACTIVE_EXP_SET, EXP_ID_SETS

def convert_mouse_tif_to_h5_single_file(mouse_name, source_dir, dest_dir):
    """
    Convert the first .tif file for a mouse to .h5 format
    """
    print(f"\nProcessing {mouse_name}...")
    
    # Find the specific .tif file for this mouse
    tif_pattern = os.path.join(source_dir, "*06MPA_50DC*.tif")
    tif_files = glob.glob(tif_pattern)
    
    if not tif_files:
        print(f"  ✗ No .tif file with pattern '*06MPA_50DC*' found in {source_dir}")
        print(f"  Available .tif files:")
        all_tif_files = glob.glob(os.path.join(source_dir, "*.tif"))
        for f in sorted(all_tif_files):
            print(f"    {os.path.basename(f)}")
        return False
    
    if len(tif_files) > 1:
        print(f"  ⚠️  Found {len(tif_files)} files matching pattern, using the first one")
    
    # Use the specific .tif file
    tif_file = tif_files[0]
    print(f"  Using file: {os.path.basename(tif_file)}")
    
    # Load the .tif file
    print(f"  Loading .tif file...")
    frames = tifffile.imread(tif_file)
    print(f"  TIFF loaded with shape: {frames.shape}")
    
    # Verify the shape is (1600, 256, 256)
    if frames.shape != (1600, 256, 256):
        print(f"  ⚠️  Warning: Expected shape (1600, 256, 256), got {frames.shape}")
    else:
        print(f"  ✓ Shape is correct: (1600, 256, 256)")
    
    # Calculate max projection for verification
    tif_max_proj = np.max(frames, axis=0)
    print(f"  Max projection shape: {tif_max_proj.shape}")
    
    # Create output .h5 file
    h5_filename = os.path.join(dest_dir, f"{mouse_name}.h5")
    print(f"  Saving to: {h5_filename}")
    
    try:
        # Save frames in HDF5 file using your method
        with h5py.File(h5_filename, 'w') as h5f:
            h5f.create_dataset('mov', data=frames, compression="gzip")
        
        # Verify the saved file
        with h5py.File(h5_filename, 'r') as h5f:
            h5_frames = h5f['mov'][:]
            h5_max_proj = np.max(h5_frames, axis=0)
        
        # Check if the data is preserved correctly
        frames_match = np.array_equal(frames, h5_frames)
        max_proj_match = np.array_equal(tif_max_proj, h5_max_proj)
        
        print(f"  ✓ Frames match: {frames_match}")
        print(f"  ✓ Max projection match: {max_proj_match}")
        print(f"  ✓ HDF5 shape: {h5_frames.shape}")
        
        file_size = os.path.getsize(h5_filename)
        print(f"  ✓ Saved: {h5_filename} ({file_size:,} bytes)")
        
        # Save max projection comparison image if matplotlib is available
        if plt is not None:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(tif_max_proj, cmap='gray')
            axs[0].set_title(f'Max Projection TIFF - {mouse_name}')
            axs[0].axis('off')
            axs[1].imshow(h5_max_proj, cmap='gray')
            axs[1].set_title(f'Max Projection HDF5 - {mouse_name}')
            axs[1].axis('off')
            plt.tight_layout()
            comparison_file = f"max_projection_comparison_{mouse_name}.png"
            plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Comparison image saved: {comparison_file}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error saving .h5 file: {e}")
        return False

def convert_folder_tifs_to_individual_h5(source_dir: str, dest_dir: str, mouse_prefix: str) -> int:
    """
    Convert every .tif in source_dir into its own .h5 file under dest_dir.
    Output filenames follow the pattern: {mouse_prefix}_{n}.h5 where n is the
    trailing number extracted from the TIFF filename (e.g., ...-781.tif → 781).
    Returns the number of successfully converted files.
    """
    if not os.path.isdir(source_dir):
        print(f"✗ Source directory not found: {source_dir}")
        return 0

    os.makedirs(dest_dir, exist_ok=True)

    tif_files = sorted(glob.glob(os.path.join(source_dir, "*.tif")))
    if not tif_files:
        print(f"✗ No .tif files found in {source_dir}")
        return 0

    print(f"Found {len(tif_files)} .tif files in {source_dir}")

    # Pattern to capture the trailing numeric token before the extension, e.g., "...-781.tif"
    trailing_num_re = re.compile(r"-(\d+)\.tif$", re.IGNORECASE)

    num_converted = 0
    for idx, tif_path in enumerate(tif_files, start=1):
        base = os.path.basename(tif_path)
        m = trailing_num_re.search(base)
        if not m:
            print(f"  ⚠️  Skipping file without trailing number: {base}")
            continue

        n_str = m.group(1)
        out_name = f"{mouse_prefix}_{n_str}.h5"
        out_path = os.path.join(dest_dir, out_name)

        if os.path.exists(out_path):
            size_bytes = os.path.getsize(out_path)
            print(f"  {out_name} already exists ({size_bytes:,} bytes) - skipping")
            num_converted += 1
            continue

        print(f"[{idx}/{len(tif_files)}] Reading {base} → saving {out_name}")
        try:
            frames = tifffile.imread(tif_path)
            with h5py.File(out_path, 'w') as h5f:
                h5f.create_dataset('mov', data=frames, compression='gzip')
            # best-effort flush verification of shape
            with h5py.File(out_path, 'r') as h5f:
                _ = h5f['mov'].shape
            print(f"    ✓ Saved {out_name}")
            num_converted += 1
        except Exception as e:
            print(f"    ✗ Failed on {base}: {e}")
            # continue with next file

    print(f"✓ Converted {num_converted}/{len(tif_files)} files to {dest_dir}")
    return num_converted

def main():
    print("Converting .tif files to .h5 format for line3_dataset (single file per mouse)")
    print("=" * 75)
    
    # Source and destination paths
    source_base = "/gpfs/data/shohamlab/nicole/tifdata"
    dest_folder = DATAFOLDER_SETS[ACTIVE_EXP_SET]
    
    # Mouse to directory mapping
    mouse_dirs = {
        'mouse6': '20191108_mouse6_region1',
        'mouse7': '20191109_mouse7_region1', 
        'mouse10': '20191110_mouse10_region1',
        'mouse12': '20191112_mouse12_region1'
    }
    
    print(f"Source base: {source_base}")
    print(f"Destination: {dest_folder}")
    print(f"Active dataset: {ACTIVE_EXP_SET}")
    
    if ACTIVE_EXP_SET != 'line3_dataset':
        print(f"Error: Expected line3_dataset, got {ACTIVE_EXP_SET}")
        return 1
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    # Convert files for each mouse
    success_count = 0
    for mouse in EXP_ID_SETS[ACTIVE_EXP_SET]:
        if mouse not in mouse_dirs:
            print(f"✗ No directory mapping found for {mouse}")
            continue
            
        source_dir = os.path.join(source_base, mouse_dirs[mouse])
        
        if not os.path.exists(source_dir):
            print(f"✗ Source directory not found: {source_dir}")
            continue
        
        # Check if file already exists
        h5_filename = os.path.join(dest_folder, f"{mouse}.h5")
        if os.path.exists(h5_filename):
            file_size = os.path.getsize(h5_filename)
            print(f"  {mouse}.h5 already exists ({file_size:,} bytes) - skipping")
            success_count += 1
            continue
        
        if convert_mouse_tif_to_h5_single_file(mouse, source_dir, dest_folder):
            success_count += 1
    
    print(f"\n✓ Conversion completed!")
    print(f"Successfully converted {success_count}/{len(EXP_ID_SETS[ACTIVE_EXP_SET])} mice")
    print(f"Check the destination folder: {dest_folder}")
    
    # List final files
    print(f"\nFinal .h5 files:")
    if os.path.exists(dest_folder):
        files = sorted([f for f in os.listdir(dest_folder) if f.endswith('.h5')])
        for f in files:
            file_path = os.path.join(dest_folder, f)
            file_size = os.path.getsize(file_path)
            print(f"  {f}: {file_size:,} bytes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
