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


##
""" Uses kineKit to generate video from catalog parameters and make videos """

# Extract info for videos to generate
df = af.get_cat_info(cat_path)

# Make the videos
af.convert_videos(df, vidin_path, vidout_path, imquality=0.75, vertpix=720)


