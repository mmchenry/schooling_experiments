##
""" Parameters and packages """
import sys
import os
import platform

# Run these lines at the iPython interpreter when developing the module code
# %load_ext autoreload
# %autoreload 2
# Use this to check version update
# af.report_version()

# These are the paths on Matt's laptop
if platform.system() == 'Darwin' and os.path.isdir('/Users/mmchenry/'):
    # Path to kineKit code
    path_kinekit = '/Users/mmchenry/Documents/code/kineKit'

    # Path to experiment catalog file
    cat_path = '/Users/mmchenry/Documents/Projects/waketracking/data/expt_catalog.csv'

    # Path to raw videos
    vidin_path = '/Users/mmchenry/Documents/Projects/waketracking/video/pilot_raw'

    # Path to exported videos
    vidout_path = '/Users/mmchenry/Documents/Projects/waketracking/video/pilot_compressed'

else:
    raise ValueError('Do not recognize this account -- add lines of code to define paths here')

# Add path to kineKit 'sources' directory using sys package
sys.path.insert(0, path_kinekit + os.path.sep + 'sources')

# Import from kineKit
import acqfunctions as af

# Extract experiment catalog info
cat = af.get_cat_info(cat_path)


##
""" Uses kineKit to crop and compress video from catalog parameters """

# Make the videos
# af.convert_videos(cat, vidin_path, vidout_path, imquality=0.75, vertpix=720)


##
""" Acquire the pixel intensity of a movie """

# import videotools as vt
import cv2 as cv  # openCV for interacting with video
import numpy as np
import pandas as pd

c_row = 0

vid_path = vidout_path + os.path.sep + cat.video_filename[c_row] + '.mp4'

# Check for file existance
if not os.path.isfile(vid_path):
    raise Exception("Video file does not exist")

# Define video object &  video frame
vid = cv.VideoCapture(vid_path)

# Video duration (in frames)
frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))

# Time step
dt = 1/cat.fps[c_row]

# Set up containers
# pix_val = pd.Series(dtype=float)
# time = pd.Series(dtype=float)

df = pd.DataFrame(columns=['time_s', 'meanpixval'])

time_c = float(0)

# Loop thru frames
for fr_num in range(1, frame_count):

    # Load image
    # vid.set(cv.CAP_PROP_POS_FRAMES, fr_num)
    # _, im = vid.read()

    df_c = pd.DataFrame([[time_c, np.mean(im, axis=(0, 1, 2))]],
                        columns=['time_s', 'meanpixval'])

    # Add to dataframe
    df = df.append(df_c, ignore_index=True)

    # Advance time
    time_c = time_c + dt


    # # Store mean pixel intensity
    # pix_val.append(np.mean(im, axis=(0, 1, 2)))
    #
    # # Store time, then advance it
    # time.append(c_time)
    # c_time = c_time + dt

# Turn off connection to video file
cv.destroyAllWindows()


