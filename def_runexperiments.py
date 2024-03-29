""" 
Functions for running experiments. Called by schooling_experiments.ipynb.
"""
#  Load packages
# from DMXEnttecPro import Controller
import time
import numpy as np
import pandas as pd
import plotly.express as px
import os
import datetime as dt
import glob
import requests

# To control the smart switch
# import asyncio
# from kasa import SmartPlug
# from unsync import unsync


def get_light_level(light_level):
    """ 
    Converts desired light level (0 to 1) to the necessary controller level to achieve that light light_level. 
    """

    control_level = 1 * light_level

    return control_level
    

def make_schedule(schedule_path, change_var=None, light_start=0.5, light_end=None, light_return=None, light_btwn=0.5,
                start_dur_min=0, end_dur_min=0, return_dur_min=0, ramp_dur_sec=0, ramp2_dur_sec=0, min_val=None, max_val=None, num_reps=1, num_trial=2, 
                btwn_period_min=5, pre_ramp_dur_sec=5, post_ramp_dur_sec=5, start_delay_min=0.5, fixed_rampdur_sec=None):
    """
    Creates a schedule of experiments to run where the value of a single variable ('change_var') 
    is changed in a random order. Variables other than 'change_var' are held constant and must be specified.

    schedule_path     - path to save the schedule
    change_var        - name of variable to change. Possible inputs: 'start_light', 'end_light', 'start_dur', 'end_dur', 'ramp_dur'.
    light_start       - light level at start of experiment (0 to 1)
    light_end         - light level at end (or middle, if third light requested) of experiment (0 to 1)
    light_return      - third light level of experiment, if requested (0 to 1)
    light_btwn        - light level between experiments (0 to 1)
    start_dur_min     - duration (min) of light level at start of experiment
    end_dur_min       - duration (min) of light level at end (or middle, if third light requested) of experiment
    return_dur_min    - duration (min) of third light level of experiment, if requested
    ramp_dur_sec      - duration (sec) of ramp between light levels
    ramp2_dur_sec      - duration (sec) of second ramp between light levels, if requested
    min_val           - minimum value of variable to change
    max_val           - maximum value of variable to change
    num_reps          - number of times to repeat each value of variable to change
    num_trial         - number of values of variable to change
    btwn_period_min   - time (min) between experiments
    pre_ramp_dur_sec  - duration of ramp before experiment starts
    post_ramp_dur_sec - duration of ramp after experiment 
    start_delay_min   - time (min) to wait before starting the experiment
    fixed_rampdur_sec - optional, additional ramp duration that is fixed and tacked on to vals after linspace generation
    """
    
    # Check inputs
    if (change_var is not None) and (change_var not in ['start_light', 'end_light', 'start_dur_min', 'end_dur_min', 'ramp_dur_sec']):
        raise ValueError("change_var must be 'start_light', 'end_light', 'start_dur_min', 'end_dur_min', 'ramp_dur_sec'")

    if change_var in ['start_light', 'end_light']:
        if min_val<0 or max_val>1:
            raise ValueError("light levels must be between 0 and 1")
        if min_val>max_val:
            raise ValueError("min_val must be less than max_val")
        if num_trial<2:
            raise ValueError("num_trial must be greater than 2")

    if change_var in ['start_dur_min', 'end_dur_min', 'ramp_dur_sec']:
        if min_val<0:
            raise ValueError("duration must be greater than 0")
        if min_val>max_val:
            raise ValueError("min_val must be less than max_val")
        if num_trial<2:
            raise ValueError("num_trial must be greater than 2")
     

    # Check ramp durations
    if (pre_ramp_dur_sec+post_ramp_dur_sec)>btwn_period_min*60:
        raise ValueError("pre_ramp_dur_sec + post_ramp_dur_sec must be less than btwn_period_min*60")
    elif pre_ramp_dur_sec>start_delay_min*60:
        raise ValueError("pre_ramp_dur_sec must be less than start_delay_min*60")

    # Identify lack of ramp
    if (change_var is None) and ((end_dur_min==0) or (ramp_dur_sec==0) or (light_end!=None)):
        print('NOTE: Creating schedule with no ramps')

        # Issue warnings
        if (end_dur_min!=0):
            print('WARNING: Igoring end_dur_min value provided')
        if (ramp_dur_sec!=0):
            print('WARNING: Igoring ramp_dur_sec value provided')
        if (light_end!=None):
            print('WARNING: Igoring light_end value provided')  

    # get current date in the format YYYY-MM-DD and define path for csv file
    date = dt.datetime.now().strftime("%Y-%m-%d")

    # Check if schedule already exists for today, add new schedule, or create new schedule
    files = glob.glob(schedule_path + os.sep + date + '*.csv')

    if len(files)>0:
        # Find largest schedule number
        last_num = 0
        for file in files:
            last_num = np.max([last_num, int(file.split('_sch')[-1].split('.')[0])])

        # Next schedule number, as a string
        nextnum_str = "{:02d}".format(last_num+1)

        # Add to schedule number for file name
        filename = date + '_sch' + nextnum_str
    else:
        filename = date + '_sch' + '01' 

    full_path = schedule_path + os.sep + filename + '.csv'

    # Make schedule dataframe with 'change_var' as a string and all other variables as floats
    schedule = pd.DataFrame(columns=['trial_num','start_time_min', 
                                     'light_start', 'light_end', 'light_return','light_btwn', 'start_dur_min', 'ramp_dur_sec',
                                     'end_dur_min','ramp2_dur_sec','return_dur_min',
                                     'pre_ramp_dur_sec', 'post_ramp_dur_sec'], dtype='object')
    
    # Make array of values to change
    if change_var is not None:
        vals = np.linspace(min_val, max_val, num_trial)
        
        if fixed_rampdur_sec!=None:
            vals = np.append(vals, fixed_rampdur_sec[0])
        
        # Repeat each value num_reps times
        vals = np.repeat(vals, num_reps)  # Repeat each value num_reps times

        # Randomize order of values
        np.random.shuffle(vals)

    else:
        vals = np.linspace(0, num_trial-1, num_trial)

    # Add non-varied variables
    for i in range(len(vals)):
        schedule.loc[i, 'trial_num']        = int(i+1)
        schedule.loc[i, 'light_start']      = light_start
        schedule.loc[i, 'light_end']        = light_end
        schedule.loc[i, 'light_return']     = light_return
        schedule.loc[i, 'light_btwn']       = light_btwn
        schedule.loc[i, 'start_dur_min']    = start_dur_min
        schedule.loc[i, 'end_dur_min']      = end_dur_min
        schedule.loc[i, 'return_dur_min']   = return_dur_min
        schedule.loc[i, 'ramp_dur_sec']     = ramp_dur_sec
        schedule.loc[i, 'ramp2_dur_sec']    = ramp2_dur_sec
        schedule.loc[i, 'pre_ramp_dur_sec'] = pre_ramp_dur_sec
        schedule.loc[i, 'post_ramp_dur_sec']= post_ramp_dur_sec

    # Overwrite values for varied variable 
    if change_var is not None:
        for i, val in enumerate(vals):
            schedule.loc[i, change_var] = val
            
            if change_var=='ramp_dur_sec' and ramp2_dur_sec!=None:
                schedule.loc[i, 'ramp2_dur_sec'] = val

    # Add start times
    for i in range(len(vals)):
        if i==0:
            schedule.loc[i, 'start_time_min'] = start_delay_min
        elif ramp2_dur_sec!=None:  
            schedule.loc[i, 'start_time_min'] = schedule.loc[i-1, 'start_time_min'] + btwn_period_min + sum(schedule.loc[i-1, ['start_dur_min', 'end_dur_min','return_dur_min']]) + sum(schedule.loc[i-1, ['ramp_dur_sec', 'ramp2_dur_sec']]/60)
            ttt=3 
        else:   
            schedule.loc[i, 'start_time_min'] = schedule.loc[i-1, 'start_time_min'] + btwn_period_min + sum(schedule.loc[i-1, ['start_dur_min', 'end_dur_min']]) + schedule.loc[i-1, 'ramp_dur_sec']/60
            ttt=3
            

    # write schedule to csv at schedule_path
    schedule.to_csv(full_path, index=False)

    # Report output
    print('--------------------------------------------------')
    print('Schedule file created: ' + full_path)
    print('--------------------------------------------------')
    print(' ')

    return filename


def run_experiment_schedule(dmx, aud_path, log_path, schedule_path, movie_prefix=None, 
                            control_hw=True, scene_num=1, shot_num=1, take_num_start=1, camera_ip=None):
    """ 
    Runs a schedule of experiments using the run_program function.
    dmx            - specifies the hardware address for the Enttex device
    aud_path       - Path to audio file to play for timecode
    log_path       - path to log file ("hardware_run_log.csv")
    schedule_path  - path to schedule file
    LED_IP         - IP address of LED controller
    movie_prefix   - prefix for movie file names
    control_hw     - True/False to control hardware
    scene_num      - scene number for movie file names
    shot_num       - shot number for movie file names
    take_num       - take number for movie file names
    """

    # Check if schedule file exists
    if not os.path.isfile(schedule_path):
        raise ValueError("Schedule file does not exist")

    # Load recording_log        
    log = pd.read_csv(log_path)

    # Read schedule file
    schedule = pd.read_csv(schedule_path)

    # Date from schedule filename
    date = schedule_path.split(os.sep)[-1].split('_sch')[0]

    # Read two digits from schedule filename that follow 'sch'
    sch_num = int(schedule_path.split(os.sep)[-1].split('_sch')[1].split('.')[0])

    # Current time (object) using 
    starttime_obj = dt.datetime.now()

    def calc_starttime(schedule, log, date, sch_num, next_trial):
        """
        Finds when to start the next trial based on the schedule and log files.
        schedule   - schedule dataframe
        log        - log dataframe
        date       - date of experiment
        next_trial - trial number for next trial
        """

        # Find start time for the last trial completed for the current date
        done_starttime_str = log.loc[(log['date']==date) & (log['sch_num']==sch_num) & (log['trial_num']==int(next_trial-1)), 'start_time'].max()
        done_starttime_obj = dt.datetime.strptime(date + ',' + done_starttime_str, "%Y-%m-%d,%H:%M:%S")

        # Find the start time for the next trial and previous trial in the schedule
        next_starttime_min  = schedule.loc[schedule['trial_num']==next_trial, 'start_time_min'].values[0]
        prev_starttime_min  = schedule.loc[schedule['trial_num']==next_trial-1, 'start_time_min'].values[0]
        next_ramp_start_sec = schedule.loc[schedule['trial_num']==next_trial, 'pre_ramp_dur_sec'].values[0]

        # Next start time is the sum of done_starttime_obj and the difference between next_starttime_obj and prev_starttime_obj
        next_starttime_obj = done_starttime_obj + dt.timedelta(minutes = (next_starttime_min - prev_starttime_min)) - dt.timedelta(seconds = float(next_ramp_start_sec))

        # if next_starttime_obj is earlier than current time, then start immediately instead
        if next_starttime_obj < dt.datetime.now():
            next_starttime_obj = starttime_obj 
            print('---- Using current time to start, rather than a future time ----')

        return next_starttime_obj

    # Check that date matches today's date
    if date != dt.datetime.now().strftime("%Y-%m-%d"):
        raise ValueError("Schedule file is not for today and the code assumes that it is.")
        # print('WARNING: Schedule file is not for today and the code assumes that it is.')

    # find match of date in log['date'].values and sch_num in log['sch_num'].values
    matching_log = log[(log['date']==date) & (log['sch_num']==sch_num)]

    # Set next trial number and start time for all experiments
    if len(matching_log) > 0:

        # Find the largest trial_num completed for the current date
        next_trial = int(log.loc[(log['date']==date) & (log['sch_num']==sch_num), 'trial_num'].max()) + 1
        
        # When to start the next experiment
        next_starttime_obj = calc_starttime(schedule, log, date, sch_num, next_trial)       

    # If this is the first experiment of the day
    else:

        # Initialize trial number to 1
        next_trial = 1

        # Time for starting the next ramp, before next experiment
        next_starttime_obj = starttime_obj + \
                            dt.timedelta(minutes = float(schedule.loc[0, 'start_time_min'])) - \
                            dt.timedelta(seconds = float(schedule.loc[0, 'pre_ramp_dur_sec']))

    #  Max trial number in schedule
    max_trial = schedule['trial_num'].max()

    # raise error if next_trial is greater than max_trial
    if next_trial > max_trial:
        raise ValueError("All trials in schedule have already been run")

    # List of trials yet to run
    trials = list(range(next_trial, max_trial+1))

    # Set initial intensity of light to light_btwn
    if control_hw:
        # DMX channel 1 to max 255
        dmx.set_channel(1, int(schedule.loc[trials[0]-1, 'light_btwn']*255))  

    # Initial take number
    take_num = take_num_start

    # Run an experiment using run_program for each in trials at the time specified in the schedule column 'start_time'
    for trial in trials:

        # Get variables from schedule for current trial
        light_start  = schedule.loc[trial-1, 'light_start']
        light_end    = schedule.loc[trial-1, 'light_end']
        light_return = schedule.loc[trial-1, 'light_return']
        light_btwn   = schedule.loc[trial-1, 'light_btwn']
        pre_dur      = schedule.loc[trial-1, 'pre_ramp_dur_sec']
        post_dur     = schedule.loc[trial-1, 'post_ramp_dur_sec']
        start_dur    = schedule.loc[trial-1, 'start_dur_min']
        end_dur      = schedule.loc[trial-1, 'end_dur_min']
        return_dur   = schedule.loc[trial-1, 'return_dur_min']
        ramp_dur     = schedule.loc[trial-1, 'ramp_dur_sec']
        ramp2_dur    = schedule.loc[trial-1, 'ramp2_dur_sec']

        # While loop to wait until current time is later than next_starttime_obj
        while dt.datetime.now() < next_starttime_obj:
            time.sleep(1)

        # Print next_starttime_obj in YYY-MM-DD format
        # print("Starting trial {} at {}".format(trial, next_starttime_obj.strftime("%Y-%m-%d, %H:%M:%S")))

        # Run pre-experiment ramp (not logged)
        run_program(dmx, aud_path, light_level=[light_btwn, light_start], light_dur=None, ramp_dur=pre_dur, 
            log_path=None, trig_video=False, echo=False, plot_data=False, movie_prefix=movie_prefix, control_hw=control_hw, sch_num=sch_num, trial_num=trial, camera_ip=camera_ip)

        # Run experiment (logged)
        run_program(dmx, aud_path, light_level=[light_start, light_end, light_return], light_dur=[start_dur, end_dur, return_dur], 
            ramp_dur=[ramp_dur, ramp2_dur], log_path=log_path, trig_video=True, echo=False, plot_data=False, movie_prefix=movie_prefix, control_hw=control_hw, scene_num=scene_num, shot_num=shot_num, take_num=take_num, sch_num=sch_num, trial_num=trial, camera_ip=camera_ip)

        if light_return==None:
        # Run post-experiment ramp (not logged)
            run_program(dmx, aud_path, light_level=[light_end, light_btwn], light_dur=None, ramp_dur=post_dur, 
                log_path=None, trig_video=False, echo=False, plot_data=False, movie_prefix=movie_prefix, control_hw=control_hw, sch_num=sch_num, trial_num=trial, camera_ip=camera_ip)
        else:
            # Run post-experiment ramp (not logged)
            run_program(dmx, aud_path, light_level=[light_return, light_btwn], light_dur=None, ramp_dur=post_dur, 
                log_path=None, trig_video=False, echo=False, plot_data=False, movie_prefix=movie_prefix, control_hw=control_hw, sch_num=sch_num, trial_num=trial, camera_ip=camera_ip)
        
        # Advance take number
        take_num = take_num + 1

        # Load latest version of log
        log = pd.read_csv(log_path)

        # When to start the next experiment
        if trial<np.max(trials):
            next_starttime_obj = calc_starttime(schedule, log, date, sch_num, trial+1)      

            # Report next start
            print("    Next: trial {} at {}".format(trial+1, next_starttime_obj.strftime("%Y-%m-%d, %H:%M:%S"))) 

    return

def control_zcam(camera_ip, command):
    """
    Controls the ZCam via HTTP commands. Assumes Ethernet connection between ZCam and computer.
    camera_ip - IP address of the ZCam
    command   - 'start' or 'stop' recording
    """

    if command == 'start':
        url = f"http://{camera_ip}/ctrl/rec?action=start"
    elif command == 'stop':
        url = f"http://{camera_ip}/ctrl/rec?action=stop"
    elif command == 'shutdown':
        url = f"http://{camera_ip}/ctrl/shutdown"
    else:
        raise ValueError("Do not recognize command: " + command)
    
    # Send request
    response = requests.get(url)
    
    # Report response
    if response.status_code == 200:
        data = response.json()
        if data['code'] == 0:
            print("ZCam successfully executed command: " + command)
        else:
            print(f"Failed to execute command (" + command + ") to ZCam. Response: {data}")
    else:
        print(f"Error: {response.status_code}")



def sync_camera_datetime(camera_ip):
    """
    Sets the date and time on the ZCam to match the computer's date and time.
    camera_ip - IP address of the ZCam
    """
    # Check communication with the camera
    test_url = f"http://{camera_ip}"
    test_response = requests.get(test_url)
    
    if test_response.status_code == 200:
        print("Communication with the camera successful. Proceeding to set date and time.")
        
        # Get current date and time from the computer
        now = dt.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Construct URL for setting date and time on the camera
        datetime_url = f"http://{camera_ip}/datetime?date={date_str}&time={time_str}"
        datetime_response = requests.get(datetime_url)
        
        if datetime_response.status_code == 200:
            data = datetime_response.json()
            if data['code'] == 0:
                print("Date and time set successfully on the camera.")
            else:
                print(f"Failed to set date and time on the camera. Response: {data}")
        else:
            print(f"Error setting date and time: {datetime_response.status_code}")
    else:
        print(f"Communication with the camera failed. Status Code: {test_response.status_code}")



def get_latest_file(camera_ip):
    """
    Reads the latest file from the ZCam.
    """
    # Step 1: List folders in /DCIM/
    folders_url = f"http://{camera_ip}/DCIM/"
    folders_response = requests.get(folders_url)
    
    if folders_response.status_code == 200:
        folders_data = folders_response.json()
        if folders_data['code'] == 0 and folders_data['files']:
            # Step 2: Choose the folder with the highest number
            latest_folder = sorted(folders_data['files'], reverse=True)[0]
            
            # Step 3: List the files in that folder
            files_url = f"http://{camera_ip}/DCIM/{latest_folder}"
            files_response = requests.get(files_url)
            
            if files_response.status_code == 200:
                files_data = files_response.json()
                if files_data['code'] == 0 and files_data['files']:
                    # Step 4: Choose the file with the latest timestamp or the highest number
                    latest_file = sorted(files_data['files'], reverse=True)[0]
                    # print(f"The latest file is: {latest_file}")
                    return latest_file
                else:
                    print(f"Failed to get files list. Response: {files_data}")
            else:
                print(f"Error listing files in folder {latest_folder}: {files_response.status_code}")
        else:
            print(f"Failed to get folders list. Response: {folders_data}")
    else:
        print(f"Error listing folders in /DCIM/: {folders_response.status_code}")

# camera_ip = "192.168.1.2"  # Replace with your camera's actual IP address
# latest_file = get_latest_file(camera_ip)


def run_program(dmx, aud_path, light_level, light_dur=None, ramp_dur=None, log_path=None, trig_video=True, 
        echo=False, plot_data=True, movie_prefix=None, control_hw=True, scene_num=None, shot_num=None, take_num=None, sch_num=999, trial_num=0, camera_ip=None):
    """ 
    Transmits signal to control light intensity via Enttex DMX USB Pro.
    dmx            - specifies the hardware address for the Enttex device
    aud_path       - Path to audio file to play for timecode
    light_level    - array of 1 to 3 values of relative light intensity levels (0 to 1)
    light_dur      - duration (in sec) that each light intensity is held fixed
    ramp_dur       - duration (in sec) of transition period between each fixed intensity level
    log_path       - path to log file ("hardware_run_log.csv")
    trig_video     - Whether to trigger the video via timecode audio
    echo           - Whether to report status througout time series
    plot_data      - whether to plot the desired timing of light changes
    movie_prefix   - text at the start of the video filenames
    control_hw     - Whether to control the hardware (if False, just logs the experiment)
    scene_num      - Scene number for video filename
    shot_num       - Shot number for video filename
    take_num       - Take number for video filename
    sch_num        - Schedule number to record to log
    trial_num      - Trial number to record to log
    """

    # Audio control described here:
    # https://stackoverflow.com/questions/57158779/how-to-stop-audio-with-playsound-module

    if control_hw and (aud_path!=None):
        import multiprocess
        from playsound import playsound

    # Check for log file
    if (log_path!=None) and (not os.path.isfile(log_path)):
        raise OSError("log_path not found at " + log_path)

    # Check for audio file
    if control_hw and trig_video and (aud_path!=None) and (not os.path.isfile(aud_path)):
        raise OSError("aud_path not found at " + aud_path)

    # Dataframe of light and control levels 
    df = make_ramp(light_level, light_dur, ramp_dur, plot_data=plot_data)

    # Current time 
    now = dt.datetime.now()

    # If using the log
    if log_path!=None:

        # Load recording_log
        log = pd.read_csv(log_path)

        # Index of latest log entry
        if len(log)==0:
            iLog = np.nan
        else:
            iLog = len(log)-1

        # Dates
        curr_date = dt.date.today()
        if not np.isnan(iLog):
            prev_date = log.date[iLog]

    # Timer starts
    starttime_str = now.strftime("%H:%M:%S")
    start_time = time.time()
    curr_time  = 0
    end_time   = max(df.time)

    # Report start
    if log_path!=None:
        print("Starting trial {} at {}".format(trial_num, starttime_str)) 

    # Start timecode audio signal
    if control_hw and trig_video and (aud_path!=None):
        p = multiprocess.Process(target=playsound, args=(aud_path, ))
        print('    Starting audio to trigger video recording')
        p.start()

    # Start video recording on ZCam
    elif control_hw and trig_video and (camera_ip!=None):
        last_filename = get_latest_file(camera_ip)
        control_zcam(camera_ip, 'start')

    # Send data to dmx in loop until time runs out
    while curr_time<end_time:          
    
        # Total time elapsed since the timer started
        curr_time = time.time() - start_time

        # Current control value (0 to 1)
        curr_control = np.interp(curr_time, df.time, df.control_level)

        if control_hw and not np.isnan(curr_control):
            # Sets DMX channel 1 in 8-bit value (Channel 1 is the intensity)
            dmx.set_channel(1, int(curr_control*255))  

        if echo:
            # Report status
            print("Time (s): " + str(round(curr_time,2)) + ", Writing to channel 1: " + str(round(curr_control,2)) )

        # Briefly pause the code to keep from overloading the hardware
        time.sleep(0.001)
        
    # End timecode audio signal
    if control_hw and trig_video and (aud_path!=None):
        p.terminate()
        print('    Timecode audio ended.')

        # Define current filename
        curr_scene   =  "{:03d}".format(scene_num)
        curr_shot    = "{:03d}".format(shot_num)
        curr_take    = "{:03d}".format(take_num)
        vid_filename = movie_prefix + '_S' + curr_scene[-3:] + '_S' + curr_shot[-3:] + '_T' + curr_take[-3:]

    # End video recording on ZCam
    elif control_hw and trig_video and (camera_ip!=None):
        control_zcam(camera_ip, 'stop')
        vid_filename = get_latest_file(camera_ip)

        # Report new file
        if vid_filename != last_filename:
            print(f"    New file created on ZCam: {vid_filename}")

    # If you are logging the ramp . . .
    if log_path!=None:

        # Data to add to log
        log_data = {
            'date': [curr_date.strftime("%Y-%m-%d")],   
            'sch_num': [int(sch_num)], 
            'trial_num' : [int(trial_num)],
            'start_time': [starttime_str],
            'video_filename': [vid_filename]
        }

        # Variable parameter inputs
        if len(light_level)>0:
            log_data['light_start']   = [light_level[0]]
            log_data['start_dur_min'] = [light_dur[0]]
        
        if len(light_level)>1:
            log_data['ramp_dur_sec'] = [ramp_dur[0]]
            log_data['light_end']    = [light_level[1]]
            log_data['end_dur_min']  = [light_dur[1]]     
        else:
            log_data['ramp_dur_sec']   = [np.nan]
            log_data['light_end']      = [np.nan]
            log_data['end_dur_min']    = [np.nan]
         

        # Append new log entry, make new indicies, save CSV log file
        log_curr = pd.DataFrame(log_data)
        log = pd.concat([log, log_curr], ignore_index=True)
        log.index = np.arange(len(log))
        log.to_csv(log_path, index=False)

        # Print results
        print("    Video filename: " + vid_filename)
        print("    Log file saved to: " + log_path)
        # State experiment
        print('    Trial ' + str(trial_num) + ' complete!')
        # print(' ')

    
def make_ramp(light_level, light_dur=None, ramp_dur=None, plot_data=False):
    """
    Generates a time series of control_level values for changes in light light_level.
    light_level - array of 1 to 3 values of relative light intensity levels (0 to 1)
    light_dur   - duration (in min) that each light intensity is held fixed
    ramp_dur    - duration (in sec) of transition period between each fixed intensity level
    plot_data   - whether to plot the desired timing of light changes
    """

    # Check inputs
    if (type(light_level)==np.ndarray) and (len(light_level)>3):
       raise ValueError("This function assumes a max of 3 light levels")
    elif (type(light_dur)==np.ndarray) and (len(light_dur)>3):
      raise ValueError("This function assumes a max of 2 light levels")
    elif (type(ramp_dur)==np.ndarray) and (len(ramp_dur)>2):
      raise ValueError("This function assumes a max of 2 ramps")
    

    # Define time step
    dt = 1/1000

    # Make empty dataframe
    df = pd.DataFrame(columns=['time','light_level','control_level'], dtype='float')   

    # No ramp
    if (type(ramp_dur)!=np.ndarray) and ramp_dur==None:

        # Check light level
        if len(light_level)>1:
            raise ValueError("Since there is no ramp, the light level should only have 1 value")
        
        # Make scalar
        if ~np.isscalar(light_level):
            light_level = light_level[0]

        # Define time vector
        tot_dur = np.sum(light_dur*60)
        df.time = np.linspace(0, tot_dur, int(round(tot_dur/dt)))

        # Initial light_level values
        df.loc[:, 'light_level'] = light_level

    # Standard ramp(s) with periods before and after
    elif (type(light_dur)==np.ndarray) or (type(light_dur)==list):

        # Define time vector
        tot_dur = np.sum(light_dur*60) + np.sum(ramp_dur)
        df.time = np.linspace(0, tot_dur, int(round(tot_dur/dt)))

        # Initial light_level values
        df.loc[df.time<=light_dur[0]*60, 'light_level'] = light_level[0]

        # Ramp
        idx = (df.time<=(light_dur[0]*60+ramp_dur[0])) & (df.time>light_dur[0]*60)
        ramp_vals = (light_level[1]-light_level[0])/ramp_dur[0] * (df.time[idx]-light_dur[0]*60) + light_level[0]
        df.loc[idx, 'light_level'] = ramp_vals

        # Second fixed light level
        idx = (df.time<=(light_dur[0]*60+ramp_dur[0]+light_dur[1]*60)) & (df.time>(light_dur[0]*60+ramp_dur[0]))
        df.loc[idx, 'light_level'] = light_level[1]     
        
        # Second ramp and third level, if necessary
        if len(light_dur)>2:

            # Second Ramp
            idx = (df.time<=(light_dur[0]*60+ramp_dur[0]+light_dur[1]*60+ramp_dur[1])) & (df.time>(light_dur[0]*60+ramp_dur[0]+light_dur[1]*60))
            ramp_vals = (light_level[2]-light_level[1])/ramp_dur[1] * (df.time[idx]-(light_dur[0]*60+ramp_dur[0]+light_dur[1]*60))+light_level[1]
            df.loc[idx, 'light_level'] = ramp_vals
            
            # Final fixed light level
            idx = (df.time<=(sum(light_dur)*60+sum(ramp_dur))) & (df.time>(light_dur[0]*60+ramp_dur[0]+light_dur[1]*60+ramp_dur[1]))
            df.loc[idx, 'light_level'] = light_level[2]

    # If just a ramp
    else:    
        # Define time vector
        tot_dur = np.sum(ramp_dur)
        df.time = np.linspace(0, tot_dur, int(round(tot_dur/dt))) 

        # Ramp
        ramp_vals = (light_level[1]-light_level[0])/ramp_dur * df.time + light_level[0]
        df.loc[:, 'light_level'] = ramp_vals
         

    # Find control level to attain each light level
    df.control_level = get_light_level(df.light_level)

    # Plot 
    if plot_data:
        fig = px.line(df, x='time', y='control_level')
        fig.show()

    return df    

