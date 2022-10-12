
#%%
""" Parameters and packages 
-----------------------------------------------------------------------------------------------------
"""
import sys
import os
import def_definepaths as dd

# Get paths (specific to system running code)
path = dd.give_paths()

# Add path to kineKit 'sources' directory using sys package
sys.path.insert(0, path['kinekit'] + os.sep + 'sources')

# Import from kineKit
import acqfunctions as af

# Extract experiment catalog info
cat = af.get_cat_info(path['cat'])

# Raw video extension
vid_ext_raw = 'MOV'

# TODO: Make ability to interactively select elliptical ROI from a single video frame
# TODO: Save PNG image to serve as mask from elliptical shape
# TODO: Add column to spreadsheet for saving mask image filename, so masks can be reused


# %% 
""" Save a single video frame to 'mask' folder
-----------------------------------------------------------------------------------------------------
"""
import videotools as vt
import cv2 as cv

# Extract experiment catalog info
cat = af.get_cat_info(path['cat'])

# Index of video in cat list to extract video
vid_index = 0

# Define path
full_path = path['vidin'] + os.sep + cat.video_filename[vid_index] + '.' + vid_ext_raw

# Extract frame and save to 'mask' directory
im = vt.get_frame(full_path)
cv.imwrite(path['mask'] + os.sep + 'frame_from_' + cat.video_filename[vid_index] + '.jpg', im)


# %% 
""" Make mask
-----------------------------------------------------------------------------------------------------
"""
import cv2 as cv
import numpy as np
import videotools as vt

# Extract experiment catalog info
cat = af.get_cat_info(path['cat'])

# Index of video in cat list to extract video
vid_index = 0

# Define path
full_path = path['vidin'] + os.sep + cat.video_filename[vid_index] + '.' + vid_ext_raw

# Extract video frame 
im = vt.get_frame(full_path)

# Extract roi coordinates
x_roi = float(cat.roi_x[vid_index])
y_roi = float(cat.roi_y[vid_index])
w_roi = float(cat.roi_w[vid_index])
h_roi = float(cat.roi_h[vid_index])
xC = x_roi + w_roi/2
yC = y_roi + h_roi/2
dims = (int(np.ceil(w_roi/2)), int(np.ceil(h_roi/2)))
cntr = (int(x_roi + w_roi/2), int(y_roi + h_roi/2))

# Define transparent image for mask
im = cv.ellipse(im, cntr, dims, angle=0, startAngle=0, endAngle=360, color=(255,255,255),thickness=-1)
trans_img = int(255/3) * np.ones((im.shape[0], im.shape[1], 4), dtype=np.uint8)
trans_img[np.where(np.all(im[..., :3] == 255, -1))] = 0

# Write mask file to disk
while True:
    filename = input('What filename do you want to use for the mask?')
    if os.path.isfile(path['mask'] + os.sep + filename + '.png'):
        print('The filename ' + filename + ' already exists. Try again.')
    else:
        cv.imwrite(path['mask'] + os.sep + filename + '.png', trans_img)
        print('Mask file saved: ' + path['mask'] + os.sep + filename + '.png')
        break


#%%
""" Uses kineKit to crop and compress video from catalog parameters 
-----------------------------------------------------------------------------------------------------
"""

# Make the masked videos (stored in 'tmp' directory)
print(' ')
print('=====================================================')
print('First, creating masked videos . . .')
af.convert_masked_videos(cat, in_path=path['vidin'], out_path=path['tmp'], maskpath=path['mask'], vmode=False, imquality=1)


# Make the downsampled/cropped videos  (stored in 'pilot_compressed' directory)
print(' ')
print('=====================================================')
print('Second, creating downsampled and cropped videos . . .')
af.convert_videos(cat, in_path=path['tmp'], out_path=path['vidout'], vmode=False, imquality=0.75, vertpix=720, suffix_in='mp4')

# Survey resulting directories 
# Loop thru each video listed in cat
print(' ')
print('=====================================================')
print('Surveying results . . .')
for c_row in cat.index:
    # Input video path
    vid_in_path = path['vidin'] + os.sep + cat.video_filename[c_row] + '.' + os.sep + vid_ext_raw

    # Temp video path
    vid_tmp_path = path['tmp'] + os.sep + cat.video_filename[c_row] + '.mp4'

    # Output video path
    vid_out_path = path['vidout'] + os.sep + cat.video_filename[c_row] + '.mp4'

    # Check that output file was made
    if not os.path.isfile(vid_out_path):

        print('   Output movie NOT created successfully: ' + vid_out_path)

        if os.path.isfile(vid_tmp_path):
            print('   Also, temp. movie NOT created successfully: ' + vid_tmp_path)
        else:
            print('   But, temp. movie created successfully: ' + vid_tmp_path)
    else:

        print('   Output movie created successfully: ' + vid_out_path)

        # Delete temp file
        if os.path.isfile(vid_tmp_path):
            os.remove(vid_tmp_path)


#%%
""" Acquire the pixel intensity from movies in cat 
-----------------------------------------------------------------------------------------------------
"""

# # import videotools as vt
# import cv2 as cv  # openCV for interacting with video
# import numpy as np
# import pandas as pd

import def_acquisition as da

# Batch run to analyze pixel intensity of all videos in cat
da.measure_pixintensity(cat, path['data'], path['vidout'])


#%%
""" Plot pixel intensity for each video analyzed 
-----------------------------------------------------------------------------------------------------
"""

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
"""" Play with parallel processing 
from https://coderzcolumn.com/tutorials/python/ipyparallel-parallel-processing-in-python
-----------------------------------------------------------------------------------------------------
"""
# At the command line, enter the following command to start a IPython Cluster (for 8 nodes):
# ipcluster start -n 8

import time
import ipyparallel as ipp
import sys
import os

print("Python Version : ", sys.version)
print("IPyparallel Version : ", ipp.__version__)

client = ipp.Client()
type(client), client.ids

load_balanced_view = client.load_balanced_view()

direct_view = client[:]

def slow_power(x, i=5):
    import time
    time.sleep(1)
    return x**i

# res = direct_view.apply(slow_power, 5,5)
# %time res.result()

# res = client[0].apply(slow_power, 4, 4)
# %time res.result()

# %time  [slow_power(i, 5) for i in range(10)]



res = direct_view.map(slow_power, range(10))
%time res.result()

# task_durations = [1] * 25
# # request a cluster
# with ipp.Cluster() as rc:
#     # get a view on the cluster
#     view = rc.load_balanced_view()
#     # submit the tasks
#     asyncresult = view.map_async(time.sleep, task_durations)
#     # wait interactively for results
#     asyncresult.wait_interactive()
#     # retrieve actual results
#     result = asyncresult.get()
# at this point, the cluster processes have been shutdown







    # Run these lines at the iPython interpreter when developing the module code
# %load_ext autoreload
# %autoreload 2
# Use this to check version update
# af.report_version()

# %%
"""" Play with parallel processing (modified)
"""

import time
import ipyparallel as ipp
import sys
import os
import pandas as pd

# print("Python Version : ", sys.version)
# print("IPyparallel Version : ", ipp.__version__)

client = ipp.Client()
type(client), client.ids

# load_balanced_view = client.load_balanced_view()

direct_view = client[:]

def slow_power(x):
    import time, os
    time.sleep(1)
    os.system('pwd')
    return x**d[x]



# direct_view["i"] = 10

direct_view["d"] = [0, 10, 12, 20, 21, 22, 33, 1]

res = []
for n in range(len(direct_view)):
    res.append(client[n].apply(slow_power, n))

[r.result() for r in res]

# res = direct_view.apply(slow_power, 5,5)
# %time res.result()

# res = client[0].apply(slow_power, 4, 4)
# %time res.result()

# %time  [slow_power(i, 5) for i in range(10)]



# res = direct_view.map(slow_power, range(10))
# %time res.result()
# %%
