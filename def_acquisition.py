""" Functions used for running the acquisition of kinematics """

import cv2 as cv  
import numpy as np
import pandas as pd
import os
import def_definepaths as dd


def measure_pixintensity(cat, data_path, vid_path):
    """Measures the mean pixel intensity for all frames in a video for all videos in a catalog,
      saves dataframe to data_path

    cat: dataframe of experiment catalog with columns that include "date", "exp_num", and "video_filename"
    data_path: path to directory where data should be saved
    vid_path: Path to video files
    """

    # Loop thru each video listed in cat
    for c_row in cat.index:

        # Unique identifier for the current sequence
        exp_name = cat.date[c_row] + '_' + format(cat.exp_num[c_row],'03')

        # Path for output data for current sequence
        dout_path = data_path + os.path.sep + exp_name + '_pixelintensity'

        # path to current video file
        vid_tot_path = vid_path + os.path.sep + cat.video_filename[c_row] + '.mp4'

        # Check for file existence
        if not os.path.isfile(vid_tot_path):
            raise Exception("Video file does not exist: " + vid_tot_path)

        # Define video object &  video frame
        vid = cv.VideoCapture(vid_tot_path)

        # Video duration (in frames)
        frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))

        # Time step between frames
        dt = 1/cat.fps[c_row]

        # Set up empty dataframe
        df = pd.DataFrame(columns=['time_s', 'meanpixval'])

        # Current time
        time_c = float(0)

        # Loop thru frames
        for fr_num in range(1, frame_count):

            # Load image
            vid.set(cv.CAP_PROP_POS_FRAMES, fr_num)
            _, im = vid.read()

            df_c = pd.DataFrame([[time_c, np.mean(im, axis=(0, 1, 2))]],
                                columns=['time_s', 'meanpixval'])

            # Add to dataframe
            df = pd.concat([df, df_c], sort=False, ignore_index=True)

            # Advance time
            time_c = time_c + dt

            print('   Pixel intensity (file ' + str(c_row+1) + ' of ' + str(len(cat)) + 
                '): frame ' + str(fr_num) + ' of ' + str(frame_count-1))

        # Write to pickle file
        df.to_pickle(dout_path)

        print('Data written to: ' +  dout_path)

            # # Store mean pixel intensity
            # pix_val.append(np.mean(im, axis=(0, 1, 2)))
            #
            # # Store time, then advance it
            # time.append(c_time)
            # c_time = c_time + dt

        # Turn off connection to video file
        cv.destroyAllWindows()


def raw_to_mat(cat):
    """ Convert all npz files for an experiment to mat files.

    cat - Catalog of experiments to convert
    """
    from scipy.io import savemat
    import glob

    # Get paths (specific to system running code)
    path = dd.give_paths() 

    # Loop thru each experiment
    for expt_c in cat.video_filename:

        # Paths for raw data files for current experiment
        path_c = path['data_raw'] + os.sep + expt_c + '*' + 'npz'

        # Get all npz filenames for current experiment
        raw_files = glob.glob(path_c)

        # Report, if no matches
        if raw_files==[]:
            print('WARNING: No raw data files match: ' + path_c)

        # Otherwise, convert
        else:
            # Loop thru each raw file
            for raw_c in raw_files:

                # Load contents of npz file
                b = np.load(raw_c)

                # Transfer all data to a dictionary
                dict_c = {}
                for field_c in b.files:
                    dict_c[field_c] = b[field_c]

                # Path for current mat file
                out_path = path['data_raw'] + os.sep + os.path.basename(raw_c)[:-4] + '.mat'

                # Save dictionary data to a mat file
                savemat(out_path, dict_c)

                # Report conversion
                print('Data export: ' + out_path)