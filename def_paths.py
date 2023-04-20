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
    if (code_path is not None) and (not os.path.exists(code_path)):
        raise Exception('Code path does not exist: ' + code_path)
  
    # add 'data' directory, if not present in proj_path
    if not os.path.exists(proj_path + os.sep + 'data'):
        os.mkdir(proj_path + os.sep + 'data')
        print('Created data directory: ' + proj_path + os.sep + 'data')

    # add 'raw' directory, if not present in proj_path + os.sep + 'data'
    if not os.path.exists(proj_path + os.sep + 'data' + os.sep + 'raw'):
        os.mkdir(proj_path + os.sep + 'data' + os.sep + 'raw')
        print('Created raw data directory: ' + proj_path + os.sep + 'data' + os.sep + 'raw')

    # add 'settings' directory, if not present in proj_path + os.sep + 'data'
    if not os.path.exists(proj_path + os.sep + 'data' + os.sep + 'settings'):
        os.mkdir(proj_path + os.sep + 'data' + os.sep + 'settings')
        print('Created settings directory: ' + proj_path + os.sep + 'data' + os.sep + 'settings')

    # add 'video' directory, if not present in proj_path
    if not os.path.exists(proj_path + os.sep + 'video'):
        os.mkdir(proj_path + os.sep + 'video')
        print('Created video directory: ' + proj_path + os.sep + 'video')

    # add 'masks' directory, if not present in proj_path
    if not os.path.exists(proj_path + os.sep + 'masks'):
        os.mkdir(proj_path + os.sep + 'masks')
        print('Created masks directory: ' + proj_path + os.sep + 'masks')

    # add 'masks' directory, if not present in proj_path
    if not os.path.exists(proj_path + os.sep + 'experiment_schedules'):
        os.mkdir(proj_path + os.sep + 'experiment_schedules')
        print('Created experiment_schedules directory: ' + proj_path + os.sep + 'experiment_schedules')

    # add 'raw' directory, if not present in proj_path + os.sep + 'video'
    if not os.path.exists(proj_path + os.sep + 'video' + os.sep + 'raw'):
        os.mkdir(proj_path + os.sep + 'video' + os.sep + 'raw')
        print('Created raw video directory: ' + proj_path + os.sep + 'video' + os.sep + 'raw')

    # add 'compressed' directory, if not present in proj_path + os.sep + 'video'
    if not os.path.exists(proj_path + os.sep + 'video' + os.sep + 'compressed'):
        os.mkdir(proj_path + os.sep + 'video' + os.sep + 'compressed')
        print('Created compressed video directory: ' + proj_path + os.sep + 'video' + os.sep + 'compressed')

    # add 'pv' directory, if not present in proj_path + os.sep + 'video'
    if not os.path.exists(proj_path + os.sep + 'video' + os.sep + 'pv'):
        os.mkdir(proj_path + os.sep + 'video' + os.sep + 'pv')
        print('Created pv video directory: ' + proj_path + os.sep + 'video' + os.sep + 'pv')

    # add 'tmp' directory, if not present in proj_path + os.sep + 'video'
    if not os.path.exists(proj_path + os.sep + 'video' + os.sep + 'tmp'):
        os.mkdir(proj_path + os.sep + 'video' + os.sep + 'tmp')
        print('Created tmp video directory: ' + proj_path + os.sep + 'video' + os.sep + 'tmp')

    # Directory structure wrt root folders
    paths = {

        # Path to experiment catalog file
        'cat': proj_path + os.sep + 'experiment_log.csv', 
        
        # Path to experiment catalog file
        'data_raw': proj_path + os.sep + 'data' + os.sep + 'raw' + os.sep ,

        # Path to settings file
        'settings': proj_path + os.sep + 'data' + os.sep + 'settings',

        # Path to raw videos
        'vidin': proj_path + os.sep + 'video' + os.sep + 'raw' + os.sep ,

        # Path to exported videos
        'vidout': proj_path + os.sep + 'video' + os.sep + 'compressed',

        # Path to exported videos
        'vidpv': proj_path + os.sep + 'video' + os.sep + 'pv',

        # Mask file
        'mask': proj_path + os.sep + 'masks',

        # Temporary video
        'tmp': proj_path + os.sep + 'video' + os.sep + 'tmp'
        }
    
    # add 'kinekit path to paths, if code_path is not None
    if code_path is not None:
        paths['kinekit'] = code_path + os.sep + 'kineKit'
    
    # Create a recording log file if it does not exist
    log_path = proj_path + os.sep + 'recording_log.csv'
    if not os.path.isfile(log_path):
        # Create an empty pandas dataframe with the column headings of 'date', 'sch_num','trail_num', write to disk
        log = pd.DataFrame(columns=['date', 'sch_num','trail_num','start_time','video_filename',
                                'analyze','light_start','light_end',
                                'start_dur_min','ramp_dur_sec','end_dur_min'])
        log.to_csv(log_path, index=False)
        print('Created recording log: ' + log_path)

    # Create an experiment log file if it does not exist
    if not os.path.isfile(paths['cat']):
        # Create an empty pandas dataframe with the column headings of 'date', 'sch_num','trail_num', write to disk
        cat = pd.DataFrame(columns=['date', 'sch_num','trail_num','school_num','fish_num','video_filename',
                                    'roi_x','roi_y','roi_w','roi_h','analyze','make_video','mask_filename',
                                    'threshold','blob_size_range','use_adaptive_threshold','adaptive_threshold_scale',
                                    'dilation_size','mask_filename','meta_real_width','settings_file','Notes'])
        cat.to_csv(paths['cat'], index=False)
        print('Created experiment log: ' + paths['cat'])

    return paths