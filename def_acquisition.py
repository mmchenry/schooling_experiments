""" Functions used for running the acquisition of kinematics """

import cv2 as cv  
import numpy as np
import pandas as pd
import os
import cv2 as cv
# import def_definepaths as dd


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
        

def make_mask(im, roi, mask_path, date, sch_num, trial_num):
    """ Make a mask for a video
    im (np.array)   - Image to mask
    roi (np.array)  - Region of interest (x, y, w, h)
    mask_path (str) - Path to save mask
    date (str)      - Date of experiment
    sch_num (int)   - Schedule number
    trial_num (int) - Trial number
    """
    # Make frame a gray field
    im = int(256/2) + 0*im

    x_roi = float(roi[0])
    y_roi = float(roi[1])
    w_roi = float(roi[2])
    h_roi = float(roi[3])

    xC = x_roi + w_roi/2
    yC = y_roi + h_roi/2
    dims = (int(np.ceil(w_roi/2)), int(np.ceil(h_roi/2)))
    cntr = (int(x_roi + w_roi/2), int(y_roi + h_roi/2))

    # Define circular image for mask
    im = cv.ellipse(im, cntr, dims, angle=0, startAngle=0, endAngle=360, color=(255,255,255),thickness=-1)

    # Start transparent image as a bunch of opaque white pixels
    trans_img = int(255)*np.ones((im.shape[0], im.shape[1], 4), dtype=np.uint8)

    # Make pixels around circle opaque gray
    trans_img[:,:,0] = int(256/2)
    trans_img[:,:,1] = int(256/2)
    trans_img[:,:,2] = int(256/2)

    # Set opacity (4th channel in image) to zero at white circle pixels in im
    trans_img[np.where(np.all(im[..., :3] == 255, -1))] = 0

    # Filename for frame of current sequence
    filename = date + '_' + format(sch_num,'03') +'_' +  format(trial_num,'03') + '_mask'

    # Output mask file
    mask_path = mask_path + os.sep + filename + '.png'

    # Write mask image to disk
    result = cv.imwrite(mask_path, trans_img)
    # result = cv.imwrite(mask_path, im)

    if result is False:
        print('Saving mask image the following path failed: ' + mask_path)
    else:
        print('Mask image saved to: ' + mask_path)


def raw_to_mat(cat, path):
    """ Convert all npz files for an experiment to mat files.

    cat - Catalog of experiments to convert
    path - Dictionary of paths to data files
    """
    from scipy.io import savemat
    import glob

    # Get paths (specific to system running code)
    # path = dd.give_paths() 

    # Loop thru each experiment
    for expt_c in cat.index:

        # Define trial filename
        trialnum = str(int(cat.trial_num[expt_c]))
        trialnum = '00' + trialnum[-3:]

        schnum = str(int(cat.sch_num[c_row]))
        schnum = '00' + schnum[-3:]

        datetrial_name = cat.date[expt_c] + '_sch' + schnum + '_tr' + trialnum
    

        # Paths for raw data files for current experiment
        path_c = path['data_raw'] + os.sep + 'data' + os.sep + datetrial_name + '*' + 'npz'

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