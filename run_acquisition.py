#%%
""" Parameters and packages """
import sys
import os
import def_definepaths as dd

# Run these lines at the iPython interpreter when developing the module code
# %load_ext autoreload
# %autoreload 2
# Use this to check version update
# af.report_version()

# Get paths (specific to system running code)
path = dd.give_paths()

# Add path to kineKit 'sources' directory using sys package
sys.path.insert(0, path['kinekit'] + os.path.sep + 'sources')

# Import from kineKit
import acqfunctions as af

# Extract experiment catalog info
cat = af.get_cat_info(cat_path)


#%%
""" Uses kineKit to crop and compress video from catalog parameters """

# Make the videos
af.convert_videos(cat, path['vidin'], path['vidout'], imquality=0.75, vertpix=720)


#%%
""" Acquire the pixel intensity from movies in cat """

# # import videotools as vt
# import cv2 as cv  # openCV for interacting with video
# import numpy as np
# import pandas as pd

import def_acquisition as da

# Batch run to analyze pixel intensity of all videos in cat
da.measure_pixintensity(cat, path['data'], path['vidout'])


#%%
""" Plot pixel intensity for each video analyzed """

import pandas as pd
import glob
import plotly.express as px


# path = os.getcwd()
csv_files = glob.glob(os.path.join(path['data'], "*.pixelintensity"))

# Loop thru each video listed in cat
for c_row in cat.index:

    # Unique identifier for the current sequence
    exp_name = cat.date[c_row] + '_' + format(cat.exp_num[c_row],'03')

    # Path for output data for current sequence
    din_path = path['data'] + os.path.sep + exp_name + '_pixelintensity'

    # Read dataframe and plot pixel intensity
    df = pd.read_pickle(din_path)
    fig = px.line(df,x="time_s", y="meanpixval", title=exp_name)
    fig.show()


# %%
