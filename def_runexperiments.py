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

# To control the smart switch
# import asyncio
# from kasa import SmartPlug
# from unsync import unsync


def get_light_level(light_level):
    """ 
    Converts desired light level (0 to 1) to the necessary controller level to achieve that light light_level. 
    """

    # TODO: measure relationship between control signal and light intensity, then update this code

    control_level = .98 * light_level

    return control_level
    

def make_schedule(schedule_path, change_var=None, light_start=None, light_end=None, light_btwn=0.5,
                start_dur_min=None, end_dur_min=None, ramp_dur_sec=None, min_val=None, max_val=None, num_val=2, 
                btwn_period_min=5, pre_ramp_dur_sec=5, post_ramp_dur_sec=5, start_delay_min=0.5):
    """
    Creates a schedule of experiments to run where the value of a single variable ('change_var') 
    is changed in a random order. Variables other than 'change_var' are held constant and must be specified.

    schedule_path     - path to save the schedule
    change_var        - name of variable to change. Possible inputs: 'start_light', 'end_light', 'start_dur', 'end_dur', 'ramp_dur'.
    light_start       - light level at start of experiment (0 to 1)
    light_end         - light level at end of experiment (0 to 1)
    light_btwn        - light level between experiments (0 to 1)
    start_dur_min     - duration (min) of light level at start of experiment
    end_dur_min       - duration (min) of light level at end of experiment
    ramp_dur_sec      - duration (sec) of ramp between light levels
    min_val           - minimum value of variable to change
    max_val           - maximum value of variable to change
    num_val           - number of values of variable to change
    btwn_period_min   - time (min) between experiments
    pre_ramp_dur_sec  - duration of ramp before experiment starts
    post_ramp_dur_sec - duration of ramp after experiment 
    start_delay_min   - time (min) to wait before starting the experiment
    """
    
    # Check inputs
    if change_var not in ['start_light', 'end_light', 'start_dur_min', 'end_dur_min', 'ramp_dur_sec']:
        raise ValueError("change_var must be 'start_light', 'end_light', 'start_dur_min', 'end_dur_min', 'ramp_dur_sec'")

    if change_var in ['start_light', 'end_light']:
        if min_val<0 or max_val>1:
            raise ValueError("light levels must be between 0 and 1")
        if min_val>max_val:
            raise ValueError("min_val must be less than max_val")
        if num_val<2:
            raise ValueError("num_val must be greater than 2")

    if change_var in ['start_dur_min', 'end_dur_min', 'ramp_dur_sec']:
        if min_val<0:
            raise ValueError("duration must be greater than 0")
        if min_val>max_val:
            raise ValueError("min_val must be less than max_val")
        if num_val<2:
            raise ValueError("num_val must be greater than 2")

    # Check ramp durations
    if (pre_ramp_dur_sec+post_ramp_dur_sec)>btwn_period_min*60:
        raise ValueError("pre_ramp_dur_sec + post_ramp_dur_sec must be less than btwn_period_min*60")
    elif pre_ramp_dur_sec>start_delay_min*60:
        raise ValueError("pre_ramp_dur_sec must be less than start_delay_min*60")

    # get current date in the format YYYY-MM-DD and define path for csv file
    date = dt.datetime.now().strftime("%Y-%m-%d")

    # Check if schedule already exists for today, add new schedule, or create new schedule
    files = glob.glob(schedule_path + os.sep + date + '*.csv')

    if len(files)>0:
        # Find largest schedule number
        last_num = 0
        for file in files:
            last_num = np.max([last_num, int(file.split('_sch')[-1].split('.')[0])])

        # Add to schedule number for file name
        filename = date + '_sch' + str(last_num + 1) 
    else:
        filename = date + '_sch' + str(1) 

    full_path = schedule_path + os.sep + filename + '.csv'

    # Make array of values to change
    vals = np.linspace(min_val, max_val, num_val)

    # Make schedule dataframe with 'change_var' as a string and all other variables as floats
    schedule = pd.DataFrame(columns=['trial_num','start_time_min', 
                                     'light_start', 'light_end', 'light_btwn', 'start_dur_min', 'ramp_dur_sec', 'end_dur_min', 
                                     'pre_ramp_dur_sec', 'post_ramp_dur_sec'], dtype='object')

    # Randomize order of values
    np.random.shuffle(vals)

    # Add non-varied variables
    for i in range(len(vals)):
        schedule.loc[i, 'trial_num']        = int(i+1)
        schedule.loc[i, 'light_start']      = light_start
        schedule.loc[i, 'light_end']        = light_end
        schedule.loc[i, 'light_btwn']       = light_btwn
        schedule.loc[i, 'start_dur_min']    = start_dur_min
        schedule.loc[i, 'ramp_dur_sec']     = schedule.loc[i, 'end_dur_min'] - schedule.loc[i, 'start_dur_min']
        schedule.loc[i, 'end_dur_min']      = end_dur_min
        schedule.loc[i, 'pre_ramp_dur_sec'] = pre_ramp_dur_sec
        schedule.loc[i, 'post_ramp_dur_sec']= post_ramp_dur_sec

    # Add values for varied variable 
    for i, val in enumerate(vals):
        schedule.loc[i, change_var] = val

    # Add start times
    for i in range(len(vals)):
        if i==0:
            schedule.loc[i, 'start_time_min'] = start_delay_min
        else:   
            schedule.loc[i, 'start_time_min'] = schedule.loc[i-1, 'start_time_min'] +      btwn_period_min + sum(schedule.loc[i-1, ['start_dur_min', 'end_dur_min']]) + schedule.loc[i-1, 'ramp_dur_sec']/60
            ttt=3

    # write schedule to csv at schedule_path
    schedule.to_csv(full_path, index=False)

    # Report output
    print('--------------------------------------------------')
    print('Schedule file created: ' + full_path)
    print('--------------------------------------------------')
    print(' ')

    return filename


def run_experiment_schedule(dmx, aud_path, log_path, schedule_path, LED_IP=None, movie_prefix=None, 
                            control_hw=True, scene_num=1, shot_num=1, take_num_start=1):
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


    # Read schedule file
    schedule = pd.read_csv(schedule_path)

    # Load recording_log
    log = pd.read_csv(log_path)

    # Date from schedule filename
    date = schedule_path.split(os.sep)[-1].split('_sch')[0]

    # Current time (object) using 
    starttime_obj = dt.datetime.now()


    def calc_starttime(schedule, log, date, next_trial):
        """
        Finds when to start the next trial based on the schedule and log files.
        schedule   - schedule dataframe
        log        - log dataframe
        date       - date of experiment
        next_trial - trial number for next trial
        """

        # Find start time for the last trial completed for the current date
        done_starttime_str = log.loc[(log['date']==date) & (log['trial_num']==next_trial-1), 'start_time'].max()
        done_starttime_obj = dt.datetime.strptime(date +',' + done_starttime_str, "%Y-%m-%d,%H:%M:%S")

        # Find the start time for the next trial and previous trial in the schedule
        next_starttime_min = schedule.loc[schedule['trial_num']==next_trial, 'start_time_min'].values[0]
        prev_starttime_min = schedule.loc[schedule['trial_num']==next_trial-1, 'start_time_min'].values[0]
        next_ramp_start_sec = schedule.loc[schedule['trial_num']==next_trial, 'pre_ramp_dur_sec'].values[0]

        # Next start time is the sum of done_starttime_obj and the difference between next_starttime_obj and prev_starttime_obj
        next_starttime_obj = done_starttime_obj + dt.timedelta(minutes = (next_starttime_min - prev_starttime_min)) - dt.timedelta(seconds = float(next_ramp_start_sec))
        
        return next_starttime_obj

    # Check that date matches today's date
    if date != dt.datetime.now().strftime("%Y-%m-%d"):
        raise ValueError("Schedule file is not for today and the code assumes that it is.")

    # Set next trial number and start time for all experiments
    if date in log['date'].values:
        # Find the largest trial_num completed for the current date
        next_trial = int(log.loc[log['date']==date, 'trial_num'].max()) + 1
        
        # When to start the next experiment
        next_starttime_obj = calc_starttime(schedule, log, date, next_trial)       

    # If this is the first experiment of the day
    else:
        # Otherwise, initialize start time
        next_trial = 1

        # Set previous end to be same as start time
        # prev_endtime_obj = starttime_obj

        # next_starttime_obj is a time object equal to the sum of starttime_obj and the first start_time_min in the schedule, minus the pre_ramp_dur_sec
        next_starttime_obj = starttime_obj + dt.timedelta(minutes = schedule.loc[0, 'start_time_min']) - dt.timedelta(seconds = float(schedule.loc[0, 'pre_ramp_dur_sec']))

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
        dmx.set_channel(1, int(schedule.loc[trials[0], 'light_btwn']*255))  

    take_num = take_num_start

    # Run an experiment using run_program for each in trials at the time specified in the schedule column 'start_time'
    for trial in trials:

        # Get variables from schedule
        light_start = schedule.loc[trial-1, 'light_start']
        light_end   = schedule.loc[trial-1, 'light_end']
        light_btwn  = schedule.loc[trial-1, 'light_btwn']
        pre_dur     = schedule.loc[trial-1, 'pre_ramp_dur_sec']
        post_dur    = schedule.loc[trial-1, 'post_ramp_dur_sec']
        start_dur   = schedule.loc[trial-1, 'start_dur_min']
        end_dur     = schedule.loc[trial-1, 'end_dur_min']
        ramp_dur    = schedule.loc[trial-1, 'ramp_dur_sec']

        # While loop to wait until current time is later than next_starttime_obj
        while dt.datetime.now() < next_starttime_obj:
            time.sleep(1)

        # Run pre-experiment ramp
        run_program(dmx, aud_path, light_level=[light_btwn, light_start], light_dur=None, ramp_dur=pre_dur, 
            log_path=None, trig_video=True, echo=False, plot_data=False, movie_prefix=movie_prefix, LED_IP=LED_IP, 
            analyze_prompt=False, control_hw=control_hw)

        # Run experiment
        run_program(dmx, aud_path, light_level=[light_start, light_end], light_dur=[start_dur, end_dur], 
            ramp_dur=ramp_dur, log_path=log_path, trig_video=True, echo=False, plot_data=False, movie_prefix=movie_prefix, LED_IP=LED_IP, 
            analyze_prompt=False, control_hw=control_hw, scene_num=scene_num, shot_num=shot_num, take_num=take_num)

        # Run post-experiment ramp
        run_program(dmx, aud_path, light_level=[light_end, light_btwn], light_dur=None, ramp_dur=post_dur, 
            log_path=None, trig_video=True, echo=False, plot_data=False, movie_prefix=movie_prefix, LED_IP=LED_IP, 
            analyze_prompt=False, control_hw=control_hw)
        
        # Advance take number
        take_num = take_num + 1

    return


def run_program(dmx, aud_path, light_level, light_dur=None, ramp_dur=None, log_path=None, trig_video=True, 
        echo=False, plot_data=True, movie_prefix=None, LED_IP=None, analyze_prompt=True, control_hw=True, 
        scene_num=1, shot_num=1, take_num=1):
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
    LED_IP         - IP address of smart switch to be controlled 
    analyze_prompt - Whether to ask whether to prompt to log the experiment
    control_hw     - Whether to control the hardware (if False, just logs the experiment)
    scene_num      - Scene number for video filename
    shot_num       - Shot number for video filename
    take_num       - Take number for video filename
    """

    # Audio control described here:
    # https://stackoverflow.com/questions/57158779/how-to-stop-audio-with-playsound-module

    if control_hw:
        import multiprocess
        from playsound import playsound

    # Check for log file
    if (log_path!=None) and (not os.path.isfile(log_path)):
        raise OSError("log_path not found at " + log_path)

    # Check for audio file
    if control_hw and (not os.path.isfile(aud_path)):
        raise OSError("aud_path not found at " + aud_path)

    # If just a ramp
    if light_dur==None:
        if len(light_level)!=2:
            raise ValueError("light_level needs to be an array of 2 values for a ramp")

    # If there is a ramp with light lights before and after
    else:
        if len(light_dur) != len(light_level):
            raise ValueError("lengths of light_level and light duration need to be equal")
        elif len(light_dur)>2:
            raise ValueError("This function assumes a max of 2 light levels")

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

        # Previous take number + 1 to new filename, if prev filename is not a nan
        if np.isnan(iLog) or pd.isnull(log.video_filename[iLog]):
            prev_take = int(take_num) - 1
        else:
            prev_take = int(log.video_filename[iLog][-3:])

        if np.isnan(iLog) or (str(prev_date) != str(curr_date.strftime("%Y-%m-%d"))):
            trial_num = 1
        else:
            trial_num = int(log.trial_num[iLog]) + 1

        # State experiment
        print('Experiment complete: ' + curr_date.strftime("%Y-%m-%d") + ', Trial number: ' + str(trial_num) + ' ----------')

    # Turn on the LEDs, wait for action to take
    if control_hw and (LED_IP is not None):
        os.system('kasa --host ' + LED_IP + ' on')
        print('    Turning on LED array')
        time.sleep(5)
   
    # Sets DMX channel 1 to max 255 (Channel 1 is the intensity)
    # dmx.set_channel(1, 255)  

    # Timer starts
    starttime_str = now.strftime("%H:%M:%S")
    start_time = time.time()
    curr_time  = 0
    end_time   = max(df.index)

    if control_hw and trig_video:
        p = multiprocess.Process(target=playsound, args=(aud_path, ))
        print('    Starting audio to trigger video recording')
        p.start()

    # Send data to dmx in loop until time runs out
    while curr_time<end_time:          
    
        # Total time elapsed since the timer started
        curr_time = time.time() - start_time

        # Current control value (0 to 1)
        curr_control = np.interp(curr_time, df.index, df.control_level)

        if control_hw:
            # Sets DMX channel 1 in 8-bit value (Channel 1 is the intensity)
            dmx.set_channel(1, int(curr_control*255))  

        if echo:
            # Report status
            print("Time (s): " + str(round(curr_time,2)) + ", Writing to channel 1: " + str(round(curr_control,2)) )

        # Briefly pause the code to keep from overloading the hardware
        time.sleep(0.001)
        
    # End timecode audio signal
    if control_hw and trig_video:
        p.terminate()
        print('    Timecode audio ended.')

    # Turn off the LEDs
    if control_hw and (LED_IP is not None):
        os.system('kasa --host ' + LED_IP + ' off')
        print('    Turning off LED array')

    # Get info about the video filename
    if (log_path!=None):
        
        # Prompt for filename numbers
        if analyze_prompt:

            prompt_txt = "Enter scene, shot, and take numbers (e.g. 1 1 " + str(int(prev_take+1)) + "):"
            scene_num, shot_num, take_num = input(prompt_txt).split()

        # Otherwise, generate them
        else:
            take_num = prev_take + 1   

    # Define current filename
    curr_scene   = '00' + str(int(scene_num))
    curr_shot    = '00' + str(int(shot_num))
    curr_take    = '00' + str(int(take_num))
    vid_filename = movie_prefix + '_S' + curr_scene[-3:] + '_S' + curr_shot[-3:] + '_T' + curr_take[-3:]

    # If you are logging the ramp . . .
    if log_path!=None:

        # Data to add to log
        log_data = {
            'date': [curr_date.strftime("%Y-%m-%d")],    
            'trial_num' : [trial_num],
            'start_time': [starttime_str],
            'video_filename': [vid_filename]
        }

        # Variable parameter inputs
        if len(light_level)>0:
            log_data['light_start']   = [light_level[0]]
            log_data['start_dur_min'] = [light_dur[0]]
        
        if len(light_level)>1:
            log_data['ramp_dur_sec'] = ramp_dur
            log_data['light_end']    = [light_level[1]]
            log_data['end_dur']      = [light_dur[1]]     
        else:
            log_data['ramp_dur_sec']   = [np.nan]
            log_data['light_end']      = [np.nan]
            log_data['end_dur_min']    = [np.nan]
         
        # Prompt and record whether to analyze recording
        if analyze_prompt:

            input_str = input("Analyze experiment [(y)es or (n)o]?")
            if input_str=='y' or input_str=='Y' or input_str=='yes' or input_str=='YES':
                log_data['analyze'] = [int(1)]
                print("    Video WILL be analyzed")
            else:
                log_data['analyze'] = [int(0)]
                print("    Video will NOT be analyzed")
        else:
            log_data['analyze'] = [int(1)]
            print("    Video WILL be analyzed")

        # Append new log entry, make new indicies, save CSV log file
        log_curr = pd.DataFrame(log_data)
        log = pd.concat([log, log_curr], ignore_index=True)
        log.index = np.arange(len(log))
        log.to_csv(log_path, index=False)

        # Print results
        print("    Video filename: " + vid_filename)
        print("    Log file saved to: " + log_path)

    
def make_ramp(light_level, light_dur=None, ramp_dur=None, plot_data=False):
    """
    Generates a time series of control_level values for changes in light light_level.
    light_level - array of 1 to 3 values of relative light intensity levels (0 to 1)
    light_dur   - duration (in sec) that each light intensity is held fixed
    ramp_dur    - duration (in sec) of transition period between each fixed intensity level
    plot_data   - whether to plot the desired timing of light changes
    """

    # Check inputs
    if (light_dur!=None) and (len(light_dur)!=2):
        raise ValueError("This function assumes 2 light levels")

    if (ramp_dur==None):
        raise ValueError("This function assumes 1 ramp duration")
    
    if len(light_level)!=2:
        raise ValueError("This function assumes a max of 2 light levels")
    
    # Standard ramp with periods before and after
    if light_dur!=None:
        # Define time vector
        dt = 1/1000
        tot_dur = np.sum(light_dur) + np.sum(ramp_dur)
        time = np.linspace(0, tot_dur, int(round(tot_dur/dt)))

        # Make empty dataframe
        df = pd.DataFrame(columns=['light_level','control_level'], index=time, dtype='float')    

        # Initial light_level values
        df.loc[df.index<=light_dur[0], 'light_level'] = light_level[0]

        # Ramp
        idx = (df.index<=(light_dur[0]+ramp_dur)) & (df.index>light_dur[0])
        ramp_vals = (light_level[1]-light_level[0])/ramp_dur * (time[idx]-light_dur[0]) + light_level[0]
        df.loc[idx, 'light_level'] = ramp_vals

        # Second fixed light level
        idx = (df.index<=(light_dur[0]+ramp_dur+light_dur[1])) & (df.index>(light_dur[0]+ramp_dur))
        df.loc[idx, 'light_level'] = light_level[1]     

    # If just a ramp
    else:    
        # Define time vector
        dt = 1/1000
        tot_dur = np.sum(ramp_dur)
        time = np.linspace(0, tot_dur, int(round(tot_dur/dt)))

        # Make empty dataframe
        df = pd.DataFrame(columns=['light_level','control_level'], index=time, dtype='float')    

        # Ramp
        ramp_vals = (light_level[1]-light_level[0])/ramp_dur * time + light_level[0]
        df.loc[:, 'light_level'] = ramp_vals
         

    # Find control level to attain each light level
    df.control_level = get_light_level(df.light_level)

    # Plot 
    if plot_data:
        fig = px.line(df, x=df.index, y='control_level')
        fig.show()

    return df    
