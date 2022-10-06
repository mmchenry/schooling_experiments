"""

function for defining the paths for all code in the project


"""

import os
import platform


def give_paths():

    # These are the paths on Matt's laptop
    if platform.system() == 'Darwin' and os.path.isdir('/Users/mmchenry/'):

        paths = {
        # Path to kineKit code
        'kinekit':  '/Users/mmchenry/Documents/code/kineKit', 

        # Path to experiment catalog file
        'cat': '/Users/mmchenry/Documents/Projects/waketracking/expt_catalog.csv', 

        # Path to experiment catalog file
        'data': '/Users/mmchenry/Documents/Projects/waketracking/data',

        # Path to raw videos
        'vidin': '/Users/mmchenry/Documents/Projects/waketracking/video/pilot_raw',

        # Path to exported videos
        'vidout': '/Users/mmchenry/Documents/Projects/waketracking/video/pilot_compressed'
        }

    elif platform.system() == 'Linux' and os.path.isdir('/home/mmchenry/'):

        paths = {
        # Path to kineKit code
        'kinekit': '/home/mmchenry/code/kineKit',

        # Path to experiment catalog file
        'cat': '/home/mmchenry/Documents/wake_tracking/expt_catalog.csv',

        # Path to output data
        'data': '/home/mmchenry/Documents/wake_tracking/data/',

        # Path to raw videos
        'vidin': '/home/mmchenry/Documents/wake_tracking/video/pilot_raw',

        # Path to exported videos
        'vidout': '/home/mmchenry/Documents/wake_tracking/video/pilot_compressed'
        }

    else:
        raise ValueError('Do not recognize this account -- add lines of code to define paths here')


    return paths