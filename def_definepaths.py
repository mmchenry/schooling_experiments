import os
import platform

def give_paths(root_code=None, root_proj=None):
    """
    function for defining the paths for all code in the project
    """

    if (root_code==None) or (root_proj==None):
        # These are the paths on Matt's laptop
        if platform.system() == 'Darwin' and os.path.isdir('/Users/mmchenry/'):

            root_code = '/Users/mmchenry/Documents/code'
            root_proj = '/Users/mmchenry/Documents/Projects/waketracking'

        # Matt on Linux
        elif platform.system() == 'Linux' and os.path.isdir('/home/mmchenry/'):

            root_code = '/home/mmchenry/code'
            root_proj = '/home/mmchenry/Documents/wake_tracking'

        # Catch alternatives
        else:
            raise ValueError('Do not recognize this account -- add lines of code to define paths here')

    # Directory structure wrt root folders
    paths = {
        # Path to kineKit code
        'kinekit':  root_code + os.sep + 'kineKit', 

        # Path to experiment catalog file
        'cat': root_proj + os.sep + 'experiment_log.csv', 

        # Path to experiment catalog file
        'data_raw': root_proj + os.sep + 'data' + os.sep + 'raw',

        # Path to settings file
        'settings': root_proj + os.sep + 'data' + os.sep + 'settings',

        # Path to raw videos
        'vidin': root_proj + os.sep + 'video' + os.sep + 'raw',

        # Path to exported videos
        'vidout': root_proj + os.sep + 'video' + os.sep + 'compressed',

        # Path to exported videos
        'vidpv': root_proj + os.sep + 'video' + os.sep + 'pv',

        # Mask file
        'mask': root_proj + os.sep + 'masks',

        # Temporary video
        'tmp': root_proj + os.sep + 'video' + os.sep + 'tmp'
        }

    return paths