""" 
Functions for running experiments. Called by schooling_experiments.ipynb.
"""
#  Load packages
from DMXEnttecPro import Controller
import time
import numpy as np
import pandas as pd
import plotly.express as px
import multiprocess
from playsound import playsound
import os
import datetime as dt

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
    

def make_ramp(light_level, light_dur, ramp_dur=0, plot_data=False):
    """
    Generates a time series of control_level values for changes in light light_level.
    light_level - array of 1 to 3 values of relative light intensity levels (0 to 1)
    light_dur   - duration (in sec) that each light intensity is held fixed
    ramp_dur    - duration (in sec) of transition period between each fixed intensity level
    plot_data   - whether to plot the desired timing of light changes
    """

    # Check inputs
    if len(light_dur) != len(light_level):
        raise ValueError("lengths of light_level and light duration need to be equal")
    elif len(light_dur)>3:
        raise ValueError("This function assumes a max of 3 light levels")

    if len(light_dur)>1 and (len(ramp_dur) != len(light_dur)-1):
        raise ValueError("length of ramp_dur should be one less than light_level and light_dur")

    # Define time vector
    dt = 1/1000
    tot_dur = np.sum(light_dur) + np.sum(ramp_dur)
    time = np.linspace(0, tot_dur, int(round(tot_dur/dt)))

    # Make empty dataframe
    df = pd.DataFrame(columns=['light_level','control_level'], index=time, dtype='float')    

    # Initial light_level values
    df.loc[df.index<=light_dur[0], 'light_level'] = light_level[0]

    # If there are any ramps
    if len(light_level)>1:
        # First ramp
        idx = (df.index<=(light_dur[0]+ramp_dur[0])) & (df.index>light_dur[0])
        ramp_vals = (light_level[1]-light_level[0])/ramp_dur[0] * (time[idx]-light_dur[0]) + light_level[0]
        df.loc[idx, 'light_level'] = ramp_vals

        # Second fixed light level
        idx = (df.index<=(light_dur[0]+ramp_dur[0]+light_dur[1])) & (df.index>(light_dur[0]+ramp_dur[0]))
        df.loc[idx, 'light_level'] = light_level[1]

        # Second ramp and third level, if necessary
        if len(light_dur)>2:

            # Second ramp
            idx = (df.index<=(light_dur[0]+ramp_dur[0]+light_dur[1]+ramp_dur[1])) & (df.index>(light_dur[0]+ramp_dur[0]+light_dur[1]))

            ramp_vals = (light_level[2]-light_level[1])/ramp_dur[1] * (time[idx]-(sum(light_dur[range(2)])+ramp_dur[0]) + light_level[1])

            df.loc[idx, 'light_level'] = ramp_vals

            # Final fixed values
            idx = df.index>(sum(light_dur[range(2)]) + sum(ramp_dur))
            df.loc[idx, 'light_level'] = light_level[2]

    # Find control level to attain each light level
    df.control_level = get_light_level(df.light_level)

    # Plot 
    if plot_data:
        fig = px.line(df, x=df.index, y='control_level')
        fig.show()
    
    return df    


def run_program(dmx, aud_path, log_path, light_level, light_dur=None, ramp_dur=0, trig_video=True, echo=False, plot_data=True, movie_prefix=None, LED_IP=None):
    """ 
    Transmits signal to control light intensity via Enttex DMX USB Pro.
    dmx          - specifies the hardware address for the Enttex device
    aud_path     - Path to audio file to play for timecode
    log_path     - path to log file ("hardware_run_log.csv")
    light_level  - array of 1 to 3 values of relative light intensity levels (0 to 1)
    light_dur    - duration (in sec) that each light intensity is held fixed
    ramp_dur     - duration (in sec) of transition period between each fixed intensity level
    trig_video   - Whether to trigger the video via timecode audio
    echo         - Whether to report status througout time series
    plot_data    - whether to plot the desired timing of light changes
    movie_prefix - text at the start of the video filenames
    LED_IP       - IP address of smart switch to be controlled 
    """

    # Audio control described here:
    # https://stackoverflow.com/questions/57158779/how-to-stop-audio-with-playsound-module

    # Check for log file
    if not os.path.isfile(log_path):
        raise OSError("log_path not found at " + log_path)

    # Check for audio file
    if not os.path.isfile(aud_path):
        raise OSError("aud_path not found at " + aud_path)

    # Check inputs
    if len(light_dur) != len(light_level):
        raise ValueError("lengths of light_level and light duration need to be equal")
    elif len(light_dur)>3:
        raise ValueError("This function assumes a max of 3 light levels")
    if len(light_dur)>1 and (len(ramp_dur) != len(light_dur)-1):
        raise ValueError("length of ramp_dur should be one less than light_level and light_dur")

    # Dataframe of light and control levels 
    df = make_ramp(light_level, light_dur, ramp_dur, plot_data=plot_data)

    # Load recording_log
    log = pd.read_csv(log_path)

    # Previous take number + 1 to new filename, if prev filename is not a nan
    if pd.isnull(log.video_filename[len(log)-1]):
        prev_take = int(1)
    else:
        prev_take = int(log.video_filename[len(log)-1][-3:])

    # Define current filename
    curr_take = '00' + str(int(prev_take+1))
    vid_filename = movie_prefix + curr_take[-3:]

    # Current time & date
    now = dt.datetime.now()
    curr_date = dt.date.today()
    prev_date = log.date[len(log)-1]

    if str(prev_date) != str(curr_date.strftime("%Y-%m-%d")):
        trial_num = 1
    else:
        trial_num = int(log.trial_num[len(log)-1]) + 1

    # State experiment
    print('Experiment -- Date: ' + curr_date.strftime("%Y-%m-%d") + ', Trial number: ' + str(trial_num) + ' ----------')

    # Turn on the LEDs, wait for action to take
    if LED_IP is not None:
        os.system('kasa --host ' + LED_IP + ' on')
        print('    Turning on LED array')
        time.sleep(1)

    # Data to add to log
    log_data = {
        'date': [curr_date.strftime("%Y-%m-%d")],    
        'trial_num' : [trial_num],
        'time': [now.strftime("%H:%M:%S")],
        'video_filename': [vid_filename]
    }

    # Variable parameter inputs
    if len(light_level)>0:
        log_data['light_level1']   = [light_level[0]]
        log_data['light_dur1']     = [light_dur[0]]
    
    if len(light_level)>1:
        log_data['ramp_dur1']      = [ramp_dur[0]]
        log_data['light_level2']   = [light_level[1]]
        log_data['light_dur2']     = [light_dur[1]]     
    else:
        log_data['ramp_dur1']      = [np.nan]
        log_data['light_level2']   = [np.nan]
        log_data['light_dur2']     = [np.nan]
         
    if len(light_level)>2:
        log_data['ramp_dur2']      = [ramp_dur[1]]
        log_data['light_level3']   = [light_level[2]]
        log_data['light_dur3']     = [light_dur[2]]  
    else:
        log_data['ramp_dur2']      = [np.nan]
        log_data['light_level3']   = [np.nan]
        log_data['light_dur3']     = [np.nan]
        
    # Sets DMX channel 1 to max 255 (Channel 1 is the intensity)
    dmx.set_channel(1, 255)  

    # Timer starts
    start_time = time.time()
    curr_time  = 0
    end_time   = max(df.index)

    if trig_video:
        p = multiprocess.Process(target=playsound, args=(aud_path, ))
        print('    Starting audio to trigger video recording')
        p.start()

    # Loop until time runs out
    while curr_time<end_time:          
    
        # Total time elapsed since the timer started
        curr_time = time.time() - start_time

        # Current control value (0 to 1)
        curr_control = np.interp(curr_time, df.index, df.control_level)

        # Sets DMX channel 1 in 8-bit value (Channel 1 is the intensity)
        dmx.set_channel(1, int(curr_control*255))  

        if echo:
            # Report status
            print("Time (s): " + str(round(curr_time,2)) + ", Writing to channel 1: " + str(round(curr_control,2)) )

        # Briefly pause the code to keep from overloading the hardware
        time.sleep(0.001)
        
    # End timecode audio signal
    if trig_video:
        p.terminate()
        print('    Timecode audio ended.')

    # Turn off the LEDs
    if LED_IP is not None:
        os.system('kasa --host ' + LED_IP + ' off')
        print('    Turning off LED array')

    # Prompt and record whether to analyze recording
    input_str = input("Analyze experiment [(y)es or (n)o]?")
    if input_str=='y' or input_str=='Y' or input_str=='yes' or input_str=='YES':
        log_data['analyze'] = [int(1)]
        print("    Video WILL be analyzed")
    else:
        log_data['analyze'] = [int(0)]
        print("    Video will NOT be analyzed")

    # Append new log entry, make new indicies, save CSV log file
    log_curr = pd.DataFrame(log_data)
    log = log.append(log_curr)
    log.index = np.arange(len(log))
    log.to_csv(log_path, index=False)

    # Print results
    print("    Video filename: " + vid_filename)
    print("    Log file saved to: " + log_path)
