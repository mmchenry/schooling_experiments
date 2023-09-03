import os
import pandas as pd
import tkinter as tk
import platform
if not (platform.system() == 'Darwin'):
    import screeninfo
import platform


def give_paths(root_path, proj_name):
    """
    Function for defining the subdirectories for all code in the project directory.
    """
    # Project path is the root path + project name
    data_path = root_path + os.sep + 'data' + os.sep + proj_name
    vid_path = root_path + os.sep + 'video' + os.sep + proj_name

    if (not os.path.exists(data_path)) and (not os.path.exists(vid_path)):
        # Show the confirmation dialog
        ans = show_confirmation_dialog('No project folders found. Create a new project in data and video folders?')

        if ans:
            # Create the project folders
            os.mkdir(data_path)
            os.mkdir(vid_path)
        else:
            # Exit the function
            return None

    elif (not os.path.exists(data_path)) and os.path.exists(vid_path):
        ans = show_confirmation_dialog('Project found in video, but not in data. Create a new project in data?')

        if ans:
            # Create the project folders
            os.mkdir(data_path)
        else:
            # Exit the function
            return None

    elif (os.path.exists(data_path)) and (not os.path.exists(vid_path)):
        ans = show_confirmation_dialog('Project found in data, but not in video. Create a new project in video?')

        if ans:
            # Create the project folders
            os.mkdir(vid_path)
        # else:
        #     # Exit the function
        #     return None

    # Directory structure wrt root folders
    paths = {
        # Path to data
        'data': data_path,

        # Path to videos
        'video': vid_path,

        # Path to experiment catalog file
        'data_raw': data_path + os.sep + 'raw',

        # Path to matlab files from TRex _results files
        'data_mat': data_path + os.sep + 'matlab',

        # Path to matlab centroid data
        'data_mat_centroid': data_path + os.sep + 'matlab' + os.sep + 'centroid',

        # Path to matlab midline data
        'data_mat_posture': data_path + os.sep + 'matlab' + os.sep + 'posture',

        # Path to settings file
        'settings': data_path + os.sep + 'settings',

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

        # Mean image file
        'mean': data_path + os.sep + 'mean_images',

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

    # Create a recording log file if it does not exist
    log_path = data_path + os.sep + 'recording_log.csv'
    if not os.path.isfile(log_path):
        # Create an empty pandas dataframe with the column headings of 'date', 'sch_num','trail_num', write to disk
        log = pd.DataFrame(columns=['date', 'sch_num','trial_num','start_time','video_filename',
                                'light_start','ramp_dur_sec','light_end',
                                'ramp2_dur_sec','light_return',
                                'start_dur_min','end_dur_min','return_dur_min'])
        log.to_csv(log_path, index=False)
        print('Created recording log: ' + log_path)

    # Create an experiment log file if it does not exist
    if not os.path.isfile(paths['cat']):
        # Create an empty pandas dataframe with the column headings of 'date', 'sch_num','trail_num', write to disk
        cat = pd.DataFrame(columns=['date','sch_num','trial_num','school_id',
                                    'exp_type','neo_treat','fish_num',
                                    'video_filename',
                                    'analyze','make_video','run_tgrabs','run_trex','run_matlab',
                                    'cm_per_pix', 'timecode_start',
                                    'threshold','min_area','max_area',
                                    'Notes'])
        cat.to_csv(paths['cat'], index=False)
        print('Created experiment log: ' + paths['cat'])

    return paths


def get_screen_resolution():
    """Get the screen resolution.
    Returns:
        width (int): Screen width.
        height (int): Screen height."""

    screen_info = screeninfo.get_monitors()
    width = screen_info[0].width
    height = screen_info[0].height
    return width, height


def show_confirmation_dialog(quest_text):

    # Linux/Windows
    if not platform.system()=='Darwin':
        # Create the main window
        root = tk.Tk()
        root.title("Confirmation")

        # Define the font for the question
        question_font = ("Arial", 40)

        # Create a label for the question
        question_label = tk.Label(root, text=quest_text, font=question_font)
        question_label.pack(padx=20, pady=20)

        # Initialize the answer variable
        answer = None

        # Function to handle the "Yes" button click
        def on_yes():
            nonlocal answer
            answer = True
            root.destroy()

        # Function to handle the "No" button click
        def on_no():
            nonlocal answer
            answer = False
            root.destroy()

        # Create the "Yes" button
        yes_button = tk.Button(root, text="Yes", font=("Arial", 24), command=on_yes)
        yes_button.pack(padx=10, pady=10)

        # Create the "No" button
        no_button = tk.Button(root, text="No", font=("Arial", 24), command=on_no)
        no_button.pack(padx=10, pady=10)

        # Get screen resolution
        screen_width, screen_height = get_screen_resolution()

        # Get the window size
        window_width = root.winfo_reqwidth()
        window_height = root.winfo_reqheight()

        # Calculate the position for the GUI window to appear centered
        position_x = int((screen_width - window_width) / 2)
        position_y = int((screen_height - window_height) / 2)

        # Set the position of the GUI window on the screen
        root.geometry(f"+{position_x}+{position_y}")

        # Start the GUI main loop
        root.mainloop()

    # For Macs
    else:
        ans_str = input(quest_text +  '(y/n)')
        if ans_str=='y':
            answer = True
        elif ans_str=='n':
            answer = False
        else:
            return None

    return answer


