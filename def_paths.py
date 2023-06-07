import os
import pandas as pd


def give_paths(root_path, proj_name, code_path=None):
    """
    Function for defining the subdirectories for all code in the project directory.
    """
    # Project path is the root path + project name
    data_path = root_path + os.sep + 'data' + os.sep + proj_name
    vid_path = root_path + os.sep + 'video' + os.sep + proj_name

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    if not os.path.exists(vid_path):
        os.mkdir(vid_path)

    #  raise exception if project paths do not exist
    # if not os.path.exists(data_path):
    #     raise Exception('Data path does not exist: ' + data_path)
    # if not os.path.exists(vid_path):
    #     raise Exception('Video path does not exist: ' + vid_path)
    
    # raise exception if code_path is not None and code_path does not exist
    # if (code_path is not None) and (not os.path.exists(code_path)):
    #     raise Exception('Code path does not exist: ' + code_path)
    
    # Directory structure wrt root folders
    paths = {
        # Path to data
        'data': data_path + os.sep + 'data',

        # Path to videos
        'video': vid_path + os.sep,

        # Path to experiment catalog file
        'data_raw': data_path + os.sep + 'data' + os.sep + 'raw',

        # Path to experiment catalog file
        'data_mat': data_path + os.sep + 'data' + os.sep + 'matlab',

        # Path to experiment catalog file
        'data_mat_vid': data_path + os.sep + 'data' + os.sep + 'matlab'+ os.sep + 'video',

        # Path to settings file
        'settings': data_path + os.sep + 'data' + os.sep + 'settings',

        # Path to raw videos
        'vidin': vid_path + os.sep + 'raw',

        # Path to exported videos
        'vidout': vid_path + os.sep + 'compressed',

        # Path to calibration videos
        'vidcal': vid_path + os.sep + 'calibration',

        # Path to pv videos
        'vidpv': vid_path + os.sep + 'pv',

        # Mask file
        'mask': data_path + os.sep + 'masks',

        # For calibration images
        'imcal': data_path + os.sep + 'calibration_images',

        # Temporary video
        'tmp': vid_path + os.sep + 'tmp',

        # Schedules
        'sch': data_path + os.sep + 'experiment_schedules'
        }
    
    # Create loop that makes a directory for each path in paths
    for path in paths.values():
        if not os.path.exists(path):
            os.mkdir(path)
            print('Created directory: ' + path)

     # Path to experiment catalog file
    paths['cat']= data_path + os.sep + 'experiment_log.csv'

    # add 'kinekit path to paths, if code_path is not None
    # if code_path is not None:
    #     paths['kinekit'] = code_path + os.sep + 'kineKit'
    
    # Create a recording log file if it does not exist
    log_path = data_path + os.sep + 'recording_log.csv'
    if not os.path.isfile(log_path):
        # Create an empty pandas dataframe with the column headings of 'date', 'sch_num','trail_num', write to disk
        log = pd.DataFrame(columns=['date', 'sch_num','trial_num','start_time','video_filename',
                                'analyze','light_start','light_end',
                                'start_dur_min','ramp_dur_sec','end_dur_min'])
        log.to_csv(log_path, index=False)
        print('Created recording log: ' + log_path)

    # Create an experiment log file if it does not exist
    if not os.path.isfile(paths['cat']):
        # Create an empty pandas dataframe with the column headings of 'date', 'sch_num','trail_num', write to disk
        cat = pd.DataFrame(columns=['date','sch_num','trial_num','school_id',
                                    'exp_type','neo_treat','fish_num',
                                    'video_filename','roi_x','roi_y','roi_w','roi_h','mask_filename',
                                    'analyze','make_video','run_matlab',
                                    'cal_video_filename','fr_width_cm','cm_per_pix',
                                    'threshold','blob_size_range',
                                    'meta_conditions','meta_species','meta_misc',
                                    'track_threshold','blob_size_ranges','track_max_speed',
                                    'meta_real_width','settings_file','Notes'])
        cat.to_csv(paths['cat'], index=False)
        print('Created experiment log: ' + paths['cat'])

    return paths