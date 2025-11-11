print('importing config')
import os

# where each dataset lives on disk
DATAFOLDER_SETS = {
    'data': '/gpfs/data/shohamlab/nicole/code/SUNS_nicole/demo/data',  # keep as-is if your demo code expects this
    'line3_dataset': '/gpfs/data/shohamlab/nicole/code/SUNS_nicole/demo/line3_dataset',  # <- outside "data", as you wanted
    'line3_scaled': '/gpfs/data/shohamlab/nicole/code/SUNS_nicole/demo/line3_scaled',  # cropped and scaled to match demo data
    'only_mouse7': '/gpfs/home/bizzin01/nicole/code/SUNS_nicole/demo/only_mouse7',  # per-video H5s for mouse7
    'mouse7_new': '/gpfs/home/bizzin01/nicole/code/SUNS_nicole/demo/mouse7_new',  # subset of only_mouse7 (first 4 videos)
    'mouse7_suite2pGT': '/gpfs/home/bizzin01/nicole/code/SUNS_nicole/demo/mouse7_suite2pGT',  # suite2p-derived GT subset
    '4video_mouse7': '/gpfs/home/bizzin01/nicole/code/SUNS_nicole/demo/4video mouse7',  # 
    '8videos_mouse7': '/gpfs/home/bizzin01/nicole/code/SUNS_nicole/demo/8videos_mouse7',  #
    '4mouse7_demo_pipeline': '/gpfs/home/bizzin01/nicole/code/SUNS_nicole_git/Shallow-UNet-Neuron-Segmentation_SUNS/demo/4mouse7_demo_pipeline',  # standard cnn + pmap start from 80
    '4mouse7_pmap': '/gpfs/home/bizzin01/nicole/code/SUNS_nicole_git/Shallow-UNet-Neuron-Segmentation_SUNS/demo/4mouse7_pmap',  # standard cnn + pmap start from 130
}
# choose which set you’re working with
ACTIVE_EXP_SET = '4mouse7_pmap'   # or 'data' / 'demo'

# identifiers used by your pipeline for each set
EXP_ID_SETS = {
    'data':   ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22'],
    'line3_dataset': ['mouse3', 'mouse6', 'mouse7', 'mouse12'],
    #'line3_dataset': ['mouse1', 'mouse3', 'mouse5', 'mouse6', 'mouse7', 'mouse9', 'mouse10', 'mouse12','mouse14'],  # include all present .h5 (added mouse10)
    # 'line3_dataset': ['mouse14'],  #leave mouse14 to test 
    'line3_scaled': ['mouse1', 'mouse3', 'mouse5', 'mouse6', 'mouse7', 'mouse9', 'mouse10', 'mouse12', 'mouse14'],  # mirror line3_dataset
    # 'only_mouse7': [
    #     'mouse7_773', 'mouse7_774', 'mouse7_775', 'mouse7_776', 'mouse7_777',
    #     'mouse7_778', 'mouse7_779', 'mouse7_780', 'mouse7_781', 'mouse7_782',
    #     'mouse7_783', 'mouse7_784', 'mouse7_785', 'mouse7_786', 'mouse7_787',
    # ],
    'only_mouse7': [
        'mouse7_773', 'mouse7_774', 'mouse7_775', 'mouse7_776', 'mouse7_777',
        'mouse7_778', 'mouse7_779', 'mouse7_780', 'mouse7_781', 'mouse7_782',
    ],
    'mouse7_new': [
        'mouse7_773', 'mouse7_774', 'mouse7_775', 'mouse7_776',
    ],

    'mouse7_suite2pGT': [
        'mouse7_773', 'mouse7_774', 'mouse7_775', 'mouse7_776',
    ],
    '4video_mouse7': [
        'mouse7_773', 'mouse7_774', 'mouse7_775', 'mouse7_776',
    ],
    '8videos_mouse7': [
        'mouse7_773', 'mouse7_774', 'mouse7_775', 'mouse7_776',
        'mouse7_777', 'mouse7_778', 'mouse7_779', 'mouse7_780',
    ],
    '4mouse7_demo_pipeline': [
        'mouse7_773', 'mouse7_774', 'mouse7_775', 'mouse7_776',
    ],
    '4mouse7_pmap': [
        'mouse7_773', 'mouse7_774', 'mouse7_775', 'mouse7_776',
    ],
}

# where to drop pipeline outputs per set
OUTPUT_FOLDER = {
    'data': 'noSF',
    'line3_dataset': 'output_line3',  # name as you like
    'line3_scaled': 'output_line3_scaled',  # outputs for scaled dataset
    'only_mouse7': 'output_8_videos',
    'mouse7_new': 'output_mouse7_new', #output_mouse7_new 
    'mouse7_suite2pGT': 'output_mouse7_suite2pGT',
    '4video_mouse7': 'output_4video_mouse7',
    '8videos_mouse7': 'output_8videos_mouse7',
    '4mouse7_demo_pipeline': 'output_4mouse7_demo_pipeline',
    '4mouse7_pmap': 'output_4mouse7_pmap',
}

# acquisition rate (Hz)
RATE_HZ = {
    'data':   10,
    'line3_dataset': 3.56,   #  from your logs
    'line3_scaled': 3.56,   # same rate as original line3_dataset
    'only_mouse7': 3.56,
    'mouse7_new': 3.56,
    'mouse7_suite2pGT': 3.56,
    '4video_mouse7': 3.56,
    '8videos_mouse7': 3.56,
    '4mouse7_demo_pipeline': 3.56,
    '4mouse7_pmap': 3.56,
}

# relative magnification vs ABO (0.785 µm/px baseline):
# MAG = 0.785 / (your_um_per_px) 
MAG = {
    'data':   6/8,  # same as demo for the original data
    'line3_dataset': 0.399,   # 0.785/1.968 if your pixel size is 2 µm/px; otherwise set 0.785/<your_um_per_px>   
    'line3_scaled': 6/8,  # same as demo data (scaled to match demo resolution)
    'only_mouse7': 0.399,
    'mouse7_new': 0.399,
    'mouse7_suite2pGT': 0.399,
    '4video_mouse7': 0.399,
    '8videos_mouse7': 0.399,
    '4mouse7_demo_pipeline': 0.399,
    '4mouse7_pmap': 0.399,
}

