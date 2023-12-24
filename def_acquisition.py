""" Functions used for running the acquisition of kinematics """

import cv2 as cv  
import numpy as np
import pandas as pd
import os
import cv2 as cv
import video_preprocess as vp
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

        # Define trial and schedule strings
        datetrial_name = generate_filename(cat.date[expt_c], cat.sch_num[expt_c], trial_num=cat.trial_num[expt_c])

        # Paths for fishdata files for current experiment
        path_c = path['data_raw'] + os.sep + 'trex_fishdata' + os.sep + datetrial_name + '*' + 'npz'

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

                # Transfer all data to a dictionary (remove #s)
                dict_c = {}
                for field_b in b.files:
                    field_c = field_b
                    field_c = field_c.replace('#', '_')
                    dict_c[field_c] = b[field_b]

                # Path for current mat file
                if len(os.path.basename(raw_c))<=35:
                    out_path = path['data_mat_centroid'] + os.sep + os.path.basename(raw_c)[:-4] + '.mat'
                else:
                    out_path = path['data_mat_posture'] + os.sep + os.path.basename(raw_c)[:-4] + '.mat'

                # Save dictionary data to a mat file
                savemat(out_path, dict_c)

                # Report conversion
                print('Data export: ' + out_path)


# Function that generates a filename
# Function that generates a filename
def generate_filename(date, sch_num, trial_num=None):
    if trial_num is None:
        return date + '_sch' + str(int(sch_num)).zfill(3)
    else:
        return date + '_sch' + str(int(sch_num)).zfill(3) + '_tr' + str(int(trial_num)).zfill(3)
    


def process_masks(npy_path):
    """
    Process npy files, find periphery coordinates of white blobs, and save to .mat files.
    
    Parameters:
        npy_path (str): Path to the directory containing npy files.
    """

    from scipy.io import savemat

    # Listing of all files with mask data
    npy_files = [file for file in os.listdir(npy_path) if file.endswith('.npy')]
    
    # Give warning if no files found
    if len(npy_files) == 0:
        print('WARNING: No npy files found in: ' + npy_path)
        
    # Loop thru each file
    for npy_file in npy_files:

        # Define matlab file
        mat_file = npy_file.replace('.npy', '.mat')

        # Define jpg file
        jpg_file = npy_file.replace('.npy', '.jpg')
        
        # If mat file does not already exists
        if mat_file not in os.listdir(npy_path):

            # Read mean image and data
            # mean_jpg  = cv2.imread(os.path.join(npy_path, jpg_file), cv2.IMREAD_GRAYSCALE)
            # mean_data = np.load(os.path.join(npy_path, npy_file))

            # Form data into the dimensions of the mean image
            # im_mask = mean_data.reshape(mean_jpg.shape[0], mean_jpg.shape[1])            
                
            # Get the mask
            # npy_data = np.load(os.path.join(npy_path, npy_file))
            im_mask, perim_mask = vp.get_mask(os.path.join(npy_path, jpg_file)) 
            
            # contours, _ = cv2.findContours(im_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # periphery_coords = contours[0][:, 0, :].tolist()
            
            savemat(os.path.join(npy_path, mat_file), {'perim_coords': perim_mask})


