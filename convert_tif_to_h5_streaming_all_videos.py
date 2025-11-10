#!/usr/bin/env python3
"""
Convert .tif files to .h5 format for line3_dataset (streaming version)
This script processes .tif files one at a time to avoid memory issues
"""

import tifffile
import h5py
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re
import argparse
import glob
import sys

# Add the suns directory to the path
sys.path.insert(0, 'suns')
from config import DATAFOLDER_SETS, ACTIVE_EXP_SET, EXP_ID_SETS

def convert_mouse_tif_to_h5_streaming(mouse_name, source_dir, dest_dir, file_pattern="*.tif"):
    """
    Convert .tif files for a mouse to .h5 format using streaming approach
    """
    print(f"\nProcessing {mouse_name}...")
    
    # Find .tif files for this mouse according to pattern
    tif_pattern = os.path.join(source_dir, file_pattern)
    tif_files = sorted(glob.glob(tif_pattern))
    
    if not tif_files:
        print(f"  ✗ No .tif files found in {source_dir}")
        return False
    
    print(f"  Found {len(tif_files)} files matching '{file_pattern}'")

    # If using a restrictive pattern (e.g., *06MPA_50DC*.tif) and multiple match, use only the first
    if file_pattern != "*.tif" and len(tif_files) > 1:
        print(f"  ⚠️  Found {len(tif_files)} matches for pattern; using the first only")
        tif_files = [tif_files[0]]
    
    # Load first .tif file to get dimensions
    print(f"  Reading first file to get dimensions...")
    first_frames = tifffile.imread(tif_files[0])
    print(f"  First file shape: {first_frames.shape}")
    
    # Calculate total frames across all files
    total_frames = 0
    for tif_file in tif_files:
        frames = tifffile.imread(tif_file)
        total_frames += frames.shape[0]
    
    print(f"  Total frames across all files: {total_frames}")
    print(f"  Expected final shape: ({total_frames}, {first_frames.shape[1]}, {first_frames.shape[2]})")
    
    # Create output .h5 file
    h5_filename = os.path.join(dest_dir, f"{mouse_name}.h5")
    print(f"  Creating .h5 file: {h5_filename}")
    
    try:
        with h5py.File(h5_filename, 'w') as h5f:
            # Create dataset with appropriate shape and chunking
            dataset = h5f.create_dataset('mov', 
                                       shape=(total_frames, first_frames.shape[1], first_frames.shape[2]),
                                       dtype=first_frames.dtype,
                                       compression='gzip',
                                       chunks=(100, first_frames.shape[1], first_frames.shape[2]))
            
            # Process each .tif file and write directly to HDF5
            frame_offset = 0
            max_proj = np.zeros((first_frames.shape[1], first_frames.shape[2]), dtype=first_frames.dtype)
            
            for i, tif_file in enumerate(tif_files):
                print(f"    Processing file {i+1}/{len(tif_files)}: {os.path.basename(tif_file)}")
                
                # Load single .tif file
                frames = tifffile.imread(tif_file)
                print(f"      Loaded {frames.shape[0]} frames")
                
                # Write frames to HDF5 dataset
                end_frame = frame_offset + frames.shape[0]
                dataset[frame_offset:end_frame] = frames
                
                # Update max projection
                file_max_proj = np.max(frames, axis=0)
                max_proj = np.maximum(max_proj, file_max_proj)
                
                # Update offset for next file
                frame_offset = end_frame
                
                # Force write to disk
                h5f.flush()
                
                # Clear frames from memory
                del frames
            
            print(f"  ✓ All frames written to HDF5")
            print(f"  Final dataset shape: {dataset.shape}")
        
        # Verify the saved file
        print(f"  Verifying saved file...")
        with h5py.File(h5_filename, 'r') as h5f:
            h5_frames = h5f['mov']
            h5_max_proj = np.max(h5_frames, axis=0)
            print(f"  HDF5 max projection shape: {h5_max_proj.shape}")
        
        # Check if max projections match
        max_proj_match = np.array_equal(max_proj, h5_max_proj)
        print(f"  ✓ Max projection match: {max_proj_match}")
        
        file_size = os.path.getsize(h5_filename)
        print(f"  ✓ Saved: {h5_filename} ({file_size:,} bytes)")
        
        # Save max projection image
        plt.figure(figsize=(8, 6))
        plt.imshow(h5_max_proj, cmap='gray')
        plt.title(f'Max Projection - {mouse_name}')
        plt.axis('off')
        
        comparison_file = f"max_projection_{mouse_name}.png"
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Max projection image saved: {comparison_file}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error saving .h5 file: {e}")
        return False

def main():
    print("Converting .tif files to .h5 format for line3_dataset (streaming version)")
    print("=" * 70)
    
    # CLI
    parser = argparse.ArgumentParser(description="Stream-convert TIFF folders to HDF5")
    parser.add_argument("--source-root", dest="source_root", default=None,
                        help="Root directory to scan for per-mouse folders (e.g., /gpfs/.../tifdata/line3)")
    parser.add_argument("--region-filter", dest="region_filter", default="region1",
                        help="Substring to require in folder names (default: region1)")
    parser.add_argument("--file-pattern", dest="file_pattern", default="*06MPA_50DC*.tif",
                        help="Glob for TIFF files inside each folder (default: *06MPA_50DC*.tif)")
    args = parser.parse_args()

    # Source and destination paths
    source_base = "/gpfs/data/shohamlab/nicole/tifdata"
    dest_folder = DATAFOLDER_SETS[ACTIVE_EXP_SET]

    print(f"Destination: {dest_folder}")
    print(f"Active dataset: {ACTIVE_EXP_SET}")

    if ACTIVE_EXP_SET != 'line3_dataset':
        print(f"Error: Expected line3_dataset, got {ACTIVE_EXP_SET}")
        return 1

    # Create destination directory if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    success_count = 0

    if args.source_root:
        # Scan mode: discover folders under source_root matching region filter
        root = args.source_root
        region_filter = (args.region_filter or "").lower()
        print(f"Scanning source root: {root}")
        print(f"Requiring folder name to contain: '{region_filter}'")

        if not os.path.isdir(root):
            print(f"✗ Source root not found: {root}")
            return 1

        candidate_dirs = []
        # Only consider immediate subdirectories for simplicity; fall back to walk if none
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if os.path.isdir(full) and region_filter in name.lower():
                candidate_dirs.append(full)
        if not candidate_dirs:
            for dirpath, dirnames, _ in os.walk(root):
                for dn in dirnames:
                    if region_filter in dn.lower():
                        candidate_dirs.append(os.path.join(dirpath, dn))

        print(f"Found {len(candidate_dirs)} candidate folders")

        mouse_regex = re.compile(r"(mouse\d+)", re.IGNORECASE)

        for source_dir in candidate_dirs:
            base = os.path.basename(source_dir)
            m = mouse_regex.search(base)
            mouse = m.group(1).lower() if m else base

            # Check if file already exists
            h5_filename = os.path.join(dest_folder, f"{mouse}.h5")
            if os.path.exists(h5_filename):
                file_size = os.path.getsize(h5_filename)
                print(f"  {mouse}.h5 already exists ({file_size:,} bytes) - skipping")
                success_count += 1
                continue

            # Optionally override file pattern at convert time by temporarily globs
            # The converter always uses *.tif; ensure there is at least one matching our pattern
            tifs = sorted(glob.glob(os.path.join(source_dir, args.file_pattern)))
            if not tifs:
                print(f"  ✗ No files matching {args.file_pattern} in {source_dir}")
                continue

            if convert_mouse_tif_to_h5_streaming(mouse, source_dir, dest_folder, args.file_pattern):
                success_count += 1

        print(f"\n✓ Conversion completed!")
        print(f"Successfully converted {success_count}/{len(candidate_dirs)} folders")
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

    # Legacy mode: use EXP_ID_SETS mapping and mouse_dirs
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
        
        if convert_mouse_tif_to_h5_streaming(mouse, source_dir, dest_folder):
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
