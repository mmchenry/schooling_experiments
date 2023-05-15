import os
import pandas as pd


def give_paths(root_path, proj_name, code_path=None):
    """
    Function for defining the subdirectories for all code in the project directory.
    """
    # Project path is the root path + project name
    proj_path = root_path + os.sep + proj_name

    #  raise exception if project path does not exist
    if not os.path.exists(proj_path):
        raise Exception('Project path does not exist: ' + proj_path)
    
    # raise exception if code_path is not None and code_path does not exist
    # if (code_path is not None) and (not os.path.exists(code_path)):
    #     raise Exception('Code path does not exist: ' + code_path)
    
    # Directory structure wrt root folders
    paths = {
        # Path to data
        'data': proj_path + os.sep + 'data',

        # Path to videos
        'video': proj_path + os.sep + 'video',

        # Path to experiment catalog file
        'data_raw': proj_path + os.sep + 'data' + os.sep + 'raw',

        # Path to experiment catalog file
        'data_mat': proj_path + os.sep + 'data' + os.sep + 'matlab',

        # Path to settings file
        'settings': proj_path + os.sep + 'data' + os.sep + 'settings',

        # Path to raw videos
        'vidin': proj_path + os.sep + 'video' + os.sep + 'raw',

        # Path to exported videos
        'vidout': proj_path + os.sep + 'video' + os.sep + 'compressed',

        # Path to exported videos
        'vidcal': proj_path + os.sep + 'video' + os.sep + 'calibration',

        # Path to exported videos
        'vidpv': proj_path + os.sep + 'video' + os.sep + 'pv',

        # Mask file
        'mask': proj_path + os.sep + 'masks',

        # For calibration images
        'imcal': proj_path + os.sep + 'calibration_images',

        # Temporary video
        'tmp': proj_path + os.sep + 'video' + os.sep + 'tmp',

        # Schedules
        'sch': proj_path + os.sep + 'experiment_schedules'
        }
    
    # Create loop that makes a directory for each path in paths
    for path in paths.values():
        if not os.path.exists(path):
            os.mkdir(path)
            print('Created directory: ' + path)

     # Path to experiment catalog file
    paths['cat']= proj_path + os.sep + 'experiment_log.csv'

    # add 'kinekit path to paths, if code_path is not None
    # if code_path is not None:
    #     paths['kinekit'] = code_path + os.sep + 'kineKit'
    
    # Create a recording log file if it does not exist
    log_path = proj_path + os.sep + 'recording_log.csv'
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
        cat = pd.DataFrame(columns=['date', 'sch_num','trial_num','school_num','fish_num','video_filename',
                                    'cal_video_filename','cm_per_pix',
                                    'roi_x','roi_y','roi_w','roi_h','analyze','make_video','mask_filename',
                                    'threshold','blob_size_range','use_adaptive_threshold','adaptive_threshold_scale',
                                    'dilation_size','mask_filename','meta_real_width','settings_file','Notes'])
        cat.to_csv(paths['cat'], index=False)
        print('Created experiment log: ' + paths['cat'])

    return paths