""" Functions used for running the acquisition of kinematics """

import videotools as vt
import cv2 as cv  # openCV for interacting with video
import os
import pandas as pd
import numpy as np
import math 
import glob


def generate_filename(date, sch_num, trial_num=None):
    """ Generates a filename for the video based on the date, schedule number, and trial number.
    Args:
        date: Date of the experiment in the format YYYYMMDD
        sch_num: Schedule number
        trial_num: Trial number (optional)
    Returns:
        filename: Filename for the video
    """
    if trial_num is None:
        return date + '_sch' + str(int(sch_num)).zfill(3)
    else:
        return date + '_sch' + str(int(sch_num)).zfill(3) + '_tr' + str(int(trial_num)).zfill(3)


def get_cat_info(cat_path, include_mode='all', exclude_mode=None, fixed_columns=None):
    """ Extracts key parameters from experiment catalog for making videos from image sequence.
    Videos included are the ones where analyze==1 and make_video==1.

    Column names must include 'date', 'trial_num', 'analyze', and 'make_video'
    
    cat_path:  Full path to video catalog (CSV file)
    include_mode: Criteria for what to include. Can be 'analyze', 'make_video', 'both', 'tgrabs', or 'trex'
    exclude_mode: Criteria for what to exclude. Can be 'calibration' or None
    fix_columns: Listing of column names to include at the start of the dataframe.

    """

    # Open CSV file
    file = open(cat_path)

    # Import CSV data
    d = pd.read_csv(file)

    # Raise error if d is empty
    if len(d) == 0:
        raise ValueError('No trial data found in experiment_log.csv.')

    change_maskname = False

    # Remove characters '_mask' or '.jpg' from mask_filename
    for i in d.index:
        entry = d.loc[i, 'mask_filename']
        
        if isinstance(entry, str):
            if '_mask' in entry:
                d.loc[i, 'mask_filename'] = entry.replace('_mask', '')
                entry = d.loc[i, 'mask_filename']
                change_maskname = True
            if '.jpg' in entry:
                d.loc[i, 'mask_filename'] = entry.replace('.jpg', '')
                change_maskname = True
        elif np.isnan(entry):
            continue
    
    if change_maskname:
         # Write d to file
        d.to_csv(cat_path, index=False)
        print('Some of the mask filenames have been updated to experiment_log.csv')

    # If fixed_columns are provided, reorder columns
    if fixed_columns is not None:

        # Sort order of non-fixed columns
        # sorted_columns = sorted([col for col in d.columns if col not in fixed_columns])
        sorted_columns = sorted([col for col in d.columns if col not in fixed_columns], key=str.casefold)
        
        # Reorder columns
        d = d[fixed_columns + sorted_columns]

        # Write d to file
        d.to_csv(cat_path, index=False)

    # Determine which rows to include
    if include_mode=='both':
        d = d.loc[(d.analyze == 1) & (d.make_video == 1)]

    elif include_mode=='analyze':
        d = d.loc[(d.analyze == 1)]
        
    elif include_mode=='make_video':
        d = d.loc[(d.make_video == 1)]
    
    elif include_mode=='tgrabs':
        d = d.loc[(d.run_tgrabs == 1)]

    elif include_mode=='trex':
        d = d.loc[(d.run_trex == 1)]

    elif include_mode=='matlab':
        d = d.loc[(d.run_matlab == 1)]
    
    elif include_mode == 'all':
        d = d

    # Determine which rows to exclude
    if exclude_mode == 'calibration':
        d = d.loc[(d.sch_num != 999)]

    # Reset indices for the new rows
    d = d.reset_index(drop=True)

    return d


def make_videos(df, im_path, vid_path, vmode=False, vertpix=None, suffix_in='JPG', suffix_out='mp4', ndigits=5, prefix='DSC', imquality=0.35):
    """ Uses videotools to create videos from the image sequences from the experiments 
    
    df: dataframe generated by get_cat_info with the info needed for each video where analyze==1 and make_video==1.
    im_path: Root directory that holds the directories named for the date of the experiment.
    vid_path: Path to directory where videos will be saved.
    vmode: Verbose mode shows more output (from ffmpeg).
    vertpix: Number of pixels in verical dimension, if downsampling. Set to None, if full resolution.
    suffix_in: Suffix for source images or movies.
    suffix_out: Suffix for output movies
    ndigits: Number of digits in input image filenames
    prefix: Prefix at the start of each image filename
    imquality: Image quality (low to high: 0 to 1) for output video
    """

    # Loop thru each video listed in df
    for c_row in df.index:

        # String for experiment number
        exp_num = '0' + str(df.exp_num[c_row])

        # Paths for current output and input videos
        vid_outpath = vid_path + os.sep + df.date[c_row] + '_' + exp_num[-2:] + '.' + suffix_out
        image_path = im_path + os.sep + df.date[c_row]

        # Read number of frames from spreadsheet
        fr_start = int(df.start_imagename[c_row][len(prefix):])
        fr_end = int(df.end_imagename[c_row][len(prefix):])


        # Match output with input frame rate
        fps = df.fps[c_row]

        # Define ROI, if needed
        roi_x = df.roi_x[c_row]
        roi_y = df.roi_y[c_row]
        roi_w = df.roi_w[c_row]
        roi_h = df.roi_h[c_row]

        if not (roi_x == 'nan'):
            r = [int(float(roi_x)), int(float(roi_y)), int(float(roi_w)), int(float(roi_h))]
        else:
            r = None

        # Create movie
        vt.vid_from_seq(image_path, vid_outpath, frStart=fr_start, frEnd=fr_end, fps=fps, imQuality=imquality,
                        prefix=prefix, nDigits=ndigits, inSuffix=suffix_in, vertPix=vertpix,
                        roi=r, vMode=vmode)

        # Report counter
        print('Finished with ' + str(c_row + 1) + ' of ' + str(len(df)) + ' videos.')


def convert_videos(df, in_path, out_path, out_name=None, in_name=None,  
    vmode=True, vertpix=None, imquality=1, suffix_in='MOV', suffix_out='mp4', 
    para_mode=False, echo=True, border_pix=None):
    """ Uses videotools to convert videos from experiments

    df: dataframe generated by get_cat_info with the info needed for each video where analyze==1 and make_video==1.
    in_path: Path to input video file (without suffix).
    out_path: Path to output file (without suffix).
    in_name: File name of input file (defaults to using filename in df), 'date_trial' specifies name as date and trial number, from those columns in df (e.g., '2022-10-01_002').
    out_name: File name of output file (defaults to same as input), 'date_trial' specifies name as date and trial number, from those columns in df.
    vmode: Verbose mode shows more output (from ffmpeg)
    imquality: Image quality (low to high: 0 to 1) for output video
    suffix_in: Suffix for source images or movies
    suffix_out: Suffix for output movies
    vertpix: Size of video frames in vertical pixels 
    para_mode: Whether to run parallel processing (requires additional code)
    echo: Whether to print the steps as they are executed
    pix_extra: Number of pixels to include around the roi
    """

    if para_mode:
        # Set up empty dataframe for parallel processing of ffmpeg commands
        cmds = pd.DataFrame(columns=['command'])

    # Loop thru each video listed in df
    for c_row in df.index:

        if border_pix is not None:
            # Define ROI without extra
            roi_x = df.roi_x[c_row] - int(np.ceil(border_pix))
            roi_y = df.roi_y[c_row] - int(np.ceil(border_pix))
            roi_w = df.roi_w[c_row] + int(np.ceil(2*border_pix))
            roi_h = df.roi_h[c_row] + int(np.ceil(2*border_pix))
        else:
            # Define ROI without extra
            roi_x = df.roi_x[c_row]
            roi_y = df.roi_y[c_row]
            roi_w = df.roi_w[c_row]
            roi_h = df.roi_h[c_row]

        if not (roi_x == 'nan'):
            r = [int(float(roi_x)), int(float(roi_y)), int(float(roi_w)), int(float(roi_h))]
        else:
            r = None

        # Overwrite vertpix
        vertpix=None

        # filename via date_trial system
        if (in_name=='date_trial') or (out_name == 'date_trial'):
            trialnum = format(int(df.trial_num[c_row]),'03')
            datetrial_name = df.date[c_row] + '_' + trialnum

        elif (in_name=='date_sch_trial') or (out_name == 'date_sch_trial'):
            trialnum = format(int(df.trial_num[c_row]),'03')
            schnum = format(int(df.sch_num[c_row]),'03')
            datetrial_name = df.date[c_row] + '_sch' + schnum + '_tr' + trialnum

        # Input path
        if (in_name=='date_trial') or (in_name=='date_sch_trial'):
            tot_in_path = in_path + os.sep + datetrial_name + '.' + suffix_in
        else:
            # Total input path (video dir path + filename)
            tot_in_path = in_path + os.sep + df.date[c_row] + os.sep + df.video_filename[c_row] + '.' + suffix_in

        # Total output path (video dir path + filename)
        if (out_name == 'date_trial') or (out_name == 'date_sch_trial'):    
            tot_out_path = out_path + os.sep + datetrial_name + '.' + suffix_out
        else:
            tot_out_path = out_path + os.sep + df.video_filename[c_row] + '.' + suffix_out     

        # Check for source video
        if not os.path.isfile(tot_in_path):
            raise OSError('Video file does not exist: ' + tot_in_path)

        # Check for output directory
        if not os.path.isdir(out_path):
            raise OSError('Output directory does not exist: ' + out_path)

        # Update status
        if echo:
            print('Converting video ' + str(c_row+1) + ' of ' + str(len(df)))

        # Create movie
        cmd = vt.vid_convert(tot_in_path, tot_out_path, imQuality=imquality, vertPix=vertpix,
                       roi=r, vMode=vmode, para_mode=para_mode, echo=echo)

        if para_mode:
            cmds_c = pd.DataFrame([[cmd]],
                                columns=['command'])

            # Add to dataframe
            cmds = pd.concat([cmds, cmds_c], sort=False, ignore_index=True)

        else:
            cmds = cmd

        # Report counter
        if echo:
            print('Finished with ' + str(c_row + 1) + ' of ' + str(len(df)) + ' videos.')

    return cmds

def convert_masked_videos(df, in_path, out_path, maskpath, in_name=None,
    out_name=None, vmode=True, imquality=1, suffix_in='MOV', 
    suffix_out='mp4', para_mode=False, echo=True):
    """ Uses videotools to convert videos from experiments

    df: dataframe generated by get_cat_info with the info needed for each video where analyze==1 and make_video==1. The code assumes that videos are in a directory named for the date (e.g. 2022-10-01) for the recording, given in the 'date' column of df.
    in_path: Path to input video file (without suffix).
    out_path: Path to output file (without suffix).
    in_name: File name of input file (defaults to using filename in df), 'date_trial' specifies name as date and trial number, from those columns in df (e.g., '2022-10-01_002').
    out_name: File name of output file (defaults to same as input), 'date_trial' specifies name as date and trial number, from those columns in df.
    vmode: Verbose mode shows more output (from ffmpeg)
    imquality: Image quality (low to high: 0 to 1) for output video
    suffix_in: Suffix for source images or movies
    suffix_out: Suffix for output movies
    maskpath: Directory path for the mask image files
    para_mode: Whether to run parallel processing (requires additional code)
    echo: Whether to print the steps as they are executed
    """
    
    if para_mode:
        # Set up empty dataframe for parallel processing of ffmpeg commands
        cmds = pd.DataFrame(columns=['command'])

    # Loop thru each video listed in df
    for c_row in df.index:

        # Total input path (video dir path + filename)
        tot_in_path = in_path + os.sep + df.date[c_row] + os.sep + df.video_filename[c_row] + '.' + suffix_in

        # filename via date_trial system
        if (in_name=='date_trial') or (out_name == 'date_trial'):
            trialnum = format(int(df.trial_num[c_row]),'03')
            datetrial_name = df.date[c_row] + '_' + trialnum

        elif (in_name=='date_sch_trial') or (out_name == 'date_sch_trial'):
            trialnum = format(int(df.trial_num[c_row]),'03')
            schnum = format(int(df.sch_num[c_row]),'03')
            datetrial_name = df.date[c_row] + '_sch' + schnum + '_tr' + trialnum

        # Input path
        if (in_name=='date_trial') or (in_name=='date_sch_trial'):
            tot_in_path = in_path + os.sep + df.date[c_row] + os.sep + datetrial_name + '.' + suffix_in
        else:
            # Total input path (video dir path + filename)
            tot_in_path = in_path + os.sep + df.date[c_row] + os.sep + df.video_filename[c_row] + '.' + suffix_in

        # Total output path (video dir path + filename)
        if (out_name == 'date_trial') or (out_name == 'date_sch_trial'):    
            tot_out_path = out_path + os.sep + datetrial_name + '.' + suffix_out
        else:
            tot_out_path = out_path + os.sep + df.video_filename[c_row] + '.' + suffix_out            

        # Total mask path 
        tot_mask_path = maskpath + os.sep + df.mask_filename[c_row] + '.png'

        if echo:
            # Update status
            print('Converting video ' + str(c_row+1) + ' of ' + str(len(df)))

        # Check for mask path
        if not os.path.isfile(tot_mask_path):
            raise OSError('Mask file does not exist: ' + tot_mask_path)

        # Check for source video
        if not os.path.isfile(tot_in_path):
            raise OSError('Video file does not exist: ' + tot_in_path)

        # Check for output directory
        if not os.path.isdir(out_path):
            raise OSError('Output directory does not exist: ' + out_path)

        # Create movie
        cmd = vt.vid_convert(tot_in_path, tot_out_path, imQuality=imquality, vMode=vmode, 
                             maskpath=tot_mask_path, para_mode=para_mode, echo=echo)

        if para_mode:
            cmds_c = pd.DataFrame([[cmd]],
                                columns=['command'])

            # Add to dataframe
            cmds = pd.concat([cmds, cmds_c], sort=False, ignore_index=True)
        else:
            cmds = cmd

        if echo:
            # Report counter
            print('Finished with ' + str(c_row + 1) + ' of ' + str(len(df)) + ' videos.')

    return cmds
  

def make_calibration_images(path, vid_ext_raw='MOV', vid_ext_mask='mp4', vid_quality=1):
    """ Makes calibration images for each calibration video in df
    
    df - dataframe generated by get_cat_info with the info needed for each video where analyze==1 and make_video==1. The code assumes that videos are in a directory named for the date (e.g. 2022-10-01) for the recording, given in the 'date' column of df.
    path - dictionary with paths to directories for input and output files
    
    """

    # Thickness (in pixels) of border around ROI
    border_pix = 10

    # Extract experiment catalog info
    df = get_cat_info(path['cat'], include_mode='make_video')
    if len(df) == 0:
        raise ValueError('No videos found in catalog file. Analyze make_video must be be set to zero for all')

    # Make unique list of cal_video_filename
    cal_video_filenames = df.cal_video_filename.unique()

    # Remove nans
    cal_video_filenames = list(filter(lambda x: isinstance(x, str) 
                                      or not math.isnan(x), cal_video_filenames))

    # Loop through each calibration video
    for cal_video_filename in cal_video_filenames:

        # Get index from catalog for first match of cal_video_filename 
        c_row = df[df.cal_video_filename == cal_video_filename].index[0]

        # Define ROI without extra
        roi_x = df.roi_x[c_row] - int(np.ceil(border_pix))
        roi_y = df.roi_y[c_row] - int(np.ceil(border_pix))
        roi_w = df.roi_w[c_row] + int(np.ceil(2*border_pix))
        roi_h = df.roi_h[c_row] + int(np.ceil(2*border_pix))

        r = [int(float(roi_x)), int(float(roi_y)), int(float(roi_w)), int(float(roi_h))]

        # Path for current calibration video
        vid_in_path = path['vidin'] + os.sep + df.date[c_row] + os.sep + cal_video_filename + '.' + vid_ext_raw

        # Define trial filename
        schnum = format(int(df.sch_num[c_row]),'03')
        trialnum = format(int(df.trial_num[c_row]),'03')
        datetrial_name = df.date[c_row] + '_sch' + schnum + '_tr' + trialnum + '_cal.' + vid_ext_mask
        
        # Paths
        tmp_path = path['tmp'] + os.sep + datetrial_name
        cal_vid_path = path['vidcal'] + os.sep + datetrial_name
        mask_path = path['mask'] + os.sep + df.mask_filename[c_row] + '.png'
        im_path =path['imcal'] + os.sep + df.mask_filename[c_row] + '.png'

        # Create movie with mask
        cmd = vt.vid_convert(vid_in_path, tmp_path, imQuality=vid_quality,
                            vMode=False, maskpath=mask_path, para_mode=False, 
                            echo=False)
        
        # Create movie with roi
        cmd = vt.vid_convert(tmp_path, cal_vid_path, imQuality=vid_quality, 
                            vertPix=None, roi=r, vMode=False, para_mode=False, echo=False)

        # Extract frame and save to 'mask' directory
        im = vt.get_frame(cal_vid_path)
        result = cv.imwrite(im_path, im)

        if result is False:
            print('Save to the following path failed: ' + im_path)
        else:
            print('Video frame saved to: ' + im_path)

def add_param_vals(cat_path, param_list_tgrabs, param_list_trex,fixed_columns=None):
    """ Adds parameter values to the catalog file
    Args:
        cat_path (str): Path to catalog file
        param_list_tgrabs (list): List of tuples with parameter name and value for tgrabs
        param_list_trex (list): List of tuples with parameter name and value for trex
    """

    # Read the full cat file
    cat = pd.read_csv(cat_path)

    # Copy the cat dataframe
    cat_start = cat.copy()

    # Add the column 'run_tgrabs' if it does not exist
    if 'run_tgrabs' not in cat.columns:
        cat['run_tgrabs'] = np.nan
    
    # Add the column 'run_trex' if it does not exist
    if 'run_trex' not in cat.columns:
        cat['run_trex'] = np.nan

    # Loop through each item in param_list_tgrabs and add a column to cat, if it does not exist
    for param_val in param_list_tgrabs:
        if param_val[0] not in cat.columns:
            cat[param_val[0]] = str(param_val[1])

        # For each row where run_tgrabs==1, set the value of the parameter to the value in param_list_tgrabs
        cat.loc[cat.run_tgrabs==1, param_val[0]] = param_val[1]

    # Loop through each item in param_list_tgrabs and add a column to cat, if it does not exist
    for param_val in param_list_trex:
        if param_val[0] not in cat.columns:
            cat[param_val[0]] = str(param_val[1])
        
        # For each row where run_tgrabs==1, set the value of the parameter to the value in param_list_tgrabs
        cat.loc[cat.run_trex==1, param_val[0]] = param_val[1]

    # Check if cat_start and cat are the same
    if cat_start.equals(cat):
        print('No new parameters added to cat file: ' + cat_path)
    else:
        # Save the cat file
        cat.to_csv(cat_path, index=False)
        print('Updated default values to cat file: ' + cat_path)

    # Reread cat to save sorted columns
    cat = get_cat_info(cat_path, include_mode='all', exclude_mode=None, fixed_columns=fixed_columns)


def run_tgrabs(cat_path, raw_path, vid_path_in, vid_path_out,  param_list_tgrabs, vid_ext_proc='mp4', use_settings_file=False,
               run_gui=True, echo=True, run_command=True, settings_path=None):
    """ Runs TGrabs on all videos listed in the catalog file
    Args:
        cat_path (str): Path to catalog file
        raw_path (str): Path to raw data (where settings files will be saved)
        vid_path_in (str): Path to input videos
        vid_path_out (str): Path to output videos
        param_list_tgrabs (list): List of tuples with parameter name and value for tgrabs
        vid_ext_proc (str): Video extension for processed videos
        use_settings_file (bool): If True, uses the settings file to run TGrabs
        run_gui (bool): If True, runs the TGrabs GUI
        echo (bool): If True, prints the command to the terminal
        run_command (bool): If True, runs the command
        settings_path (str): Path to settings file
    Returns:
        commands (list): List of commands
    """

    
    if use_settings_file:
        # Define input parameter list for TGrabs as dataframe
        param_input = pd.DataFrame(columns=['param_name', 'param_val'])
        # Add the TRex parameter listing to the TGrabs parameters (might improve the preliminary tracking)
        param_input.append(param_list_tgrabs)
        param_input.append(param_list_trex)

    # Extract experiment catalog info
    cat_curr = get_cat_info(cat_path, include_mode='tgrabs', exclude_mode='calibration')
    if len(cat_curr) == 0:
        raise ValueError('No tgrabs to work on from experiment_log.' + \
                         ' The column \'run_tgrabs\' must be' + \
                         ' set to 1 to run tgrabs.')

    # Make list of .mp4 files in local_path + os.sep + 'binary'
    binary_list = glob.glob(vid_path_in + os.sep + '*.mp4')

    commands = []

    # Loop thru each video listed in cat
    for c_row in cat_curr.index:

        # Get date, trial, and schedule numbers
        date_curr   = cat_curr.date[c_row]
        trial_curr  = cat_curr.trial_num[c_row]
        sch_curr    = cat_curr.sch_num[c_row]

        # Generic filename for the trial
        filename = generate_filename(date_curr, sch_curr, trial_num=trial_curr)

        # Define and check input path
        path_in = vid_path_in + os.sep + filename + '.' + vid_ext_proc

        # Report if there is no binary file
        if not os.path.exists(path_in):
            print(' ')
            print('No binary file for this trial, cannot generate pv file: ' + filename)

        # Otherwise, proceed with tGrabs . . .
        else:
            
            # Write settings file
            settings_path = raw_path + os.sep + filename + '.settings'
            #param_input.to_csv(settings_path, index=False)

            # Output path
            path_out = vid_path_out + os.sep + filename + '.pv'

            # Start formulating the TGrabs command
            command = f'tgrabs -i {path_in} -o {path_out} '

            # Whether to launch gui
            if run_gui:
                command += '-nowindow false '
            else:
                command += '-nowindow true '

            # Get max number of fish from cat_curr
            command += f'-track_max_individuals {str(int(cat_curr.fish_num[c_row]))} '

            # loop thru each entry in param_list_tgrabs, and add the value for that parameter in the column of cat_curr
            # that has the same name as the parameter (nans excluded)
            for param in param_list_tgrabs:
                value = cat_curr[param[0]][c_row]
                if pd.notna(value) and value != 'nan' and value != 'null':
                    command += f'-{param[0]} {value} '

            if use_settings_file:
                # Write settings to csv
                param_input.to_csv(path_settings, index=False)

                # Path to save settings table
                path_settings = settings_path + os.sep + filename + '_tgrabs_settings.csv'

            if run_command:
                # Execute at th e command line
                os.system(command)

                if echo:
                    print(' ')
                    print('Running TGrabs with the following command:')
                    print(command)
            else:
                if echo:
                    print(' ')
                    print('TGrabs command (not run):')
                    print(command)

            # Append command to list
            commands.append(command)

    return commands

def run_trex(cat_path, vid_path, data_path, param_list_trex, cat_to_trex, use_settings_file=False, output_posture=True,
             echo=True, run_gui=True, auto_quit=False, run_command=True, settings_path=None):
    """ Runs TRex on all videos listed in the catalog file
    Args:
        cat_path (str): Path to catalog file
        vid_path (str): Path to videos
        data_path (str): Path to data
        param_list_trex (list): List of tuples with parameter name and value for trex
        cat_to_trex (list): List of tuples with catalog column name and trex parameter name
        use_settings_file (bool): If True, uses the settings file to run TRex
        echo (bool): If True, prints the command to the terminal
        run_gui (bool): If True, runs the TRex GUI
        run_command (bool): If True, runs the command
        settings_path (str): Path to settings file
    Returns:
        commands (list): List of commands
    """

    # Parameter list for TGrabs/TRex
    param_input = pd.DataFrame(columns=['param_name', 'param_val'])

    if use_settings_file:
        # Define input parameter list for TGrabs as dataframe
        param_input.append(param_list_tgrabs)

        # Add the TRex parameter listing to the TGrabs parameters (might improve the preliminary tracking)
        param_input.append(param_list_trex)

    # Extract experiment catalog info
    cat_curr = get_cat_info(cat_path, include_mode='trex', exclude_mode='calibration')
    if len(cat_curr) == 0:
        raise ValueError('No tRex to work on from experiment_log.' + \
                         ' The column \'run_trex\' must be' + \
                         ' set to 1 to run tRex.')

    # Make list of .mp4 files in local_path + os.sep + 'binary'
    pv_list = glob.glob(vid_path + os.sep + '*.pv')

    commands = []

    # Loop thru each video listed in cat
    for c_row in cat_curr.index:

        # Get date, trial, and schedule numbers
        date_curr   = cat_curr.date[c_row]
        trial_curr  = cat_curr.trial_num[c_row]
        sch_curr    = cat_curr.sch_num[c_row]

        # Generic filename for the trial
        filename = generate_filename(date_curr, sch_curr, trial_num=trial_curr)

        # Define and check input path
        path_in = vid_path + os.sep + filename + '.pv'
        print(path_in)
        # Report if there is no binary file
        if not os.path.exists(path_in):
            print(' ')
            print('No pv file for this trial, cannot generate tracking data: ' + filename)

        # Otherwise, proceed with tGrabs . . .
        else:

            # Start formulating the TRex command
            settings_path = data_path + os.sep + filename + '.settings'
            command = f'trex -i {path_in} -output_dir {data_path} -s {settings_path} '

            # Whether to launch gui
            if run_gui:
                command += '-nowindow false '
                
                # Whether to auto_quit when launching qui (useful for observation)
                if auto_quit:
                    command += '-auto_quit true '
                else:    
                    command += '-auto_quit false '


            else:
                command += '-nowindow true '
                command += '-auto_quit true '

            if output_posture:
                command += '-output_posture_data true '
            else:
                command += '-output_posture_data false '    

            # Add additional default parameters
            command += '-fishdata_dir \'trex_fishdata\' '
            command += '-output_invalid_value nan '

            # Loop trhu cat_to_trexand add the value in cat to the command
            for param in cat_to_trex:
                value = cat_curr[param[0]][c_row]
                if pd.notna(value) and value != 'nan' and value != 'null':
                    command += f'-{param[1]} {value} '
                else:
                    raise ValueError('The column ' + param[0] + ' must be defined in experiment_log for each trial.')

            # loop thru each entry in param_list_tgrabs, and add the value for that parameter in the column of cat_curr
            # that has the same name as the parameter (nans excluded)
            for param in param_list_trex:
                value = str(cat_curr[param[0]][c_row]).lower()
                if pd.notna(value) and value != 'nan' and value != 'null':
                    command += f'-{param[0]} {value} '

            if use_settings_file:
                # Write settings to csv
                param_input.to_csv(path_settings, index=False)

                # Settings path
                path_settings = settings_path + os.sep + filename + '.settings'

            if run_command:
                # Execute at th e command line
                os.system(command)

                if echo:
                    print(' ')
                    print('Running TRex with the following command:')
                    print(command)
            else:
                if echo:
                    print(' ')
                    print('TRex command (not run):')
                    print(command)

            # Append command to list
            commands.append(command)

    return commands

def delete_matching_files(local_path, pv_path):
    mp4_files = [os.path.splitext(file)[0] for file in os.listdir(local_path) if file.endswith(".mp4")]
    pv_files = [os.path.splitext(file)[0] for file in os.listdir(pv_path) if file.endswith(".pv")]

    matching_files = set(mp4_files).intersection(pv_files)

    for file in matching_files:
        mp4_file_path = os.path.join(local_path, file + ".mp4")
        os.remove(mp4_file_path)
        print(f"Deleted: {mp4_file_path}")