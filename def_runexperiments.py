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


def get_light_level(light_level):
    """ 
    Converts desired light level (0 to 1) to the necessary controller level to achieve that light light_level. 
    """

    # TODO: measure relationship between control signal and light intensity, then update this code

    control_level = .98 * light_level

    return control_level
    

def make_ramp(light_level, light_dur, ramp_dur, plot_data=False):
    """
    Generates a time series of control_level values for changes in light light_level.

    light_level   - array of 2 to 3 values of relative light intensity levels (0 to 1)
    light_dur   - duration (in sec) that each light intensity is held fixed
    ramp_dur    - duration (in sec) of transition period between each fixed intensity level
    plot_data   - whether to plot the desired timing of light changes
    """

    # Check inputs
    if len(light_dur) != len(light_level):
        raise ValueError("lengths of light_level and light duration need to be equal")
    elif len(ramp_dur) != len(light_dur)-1:
        raise ValueError("length of ramp_dur should be one less than light_level and light_dur")
    elif len(light_dur)>3:
        raise ValueError("This function assumes a max of 3 light levels")
    elif len(light_dur)<2:
        raise ValueError("This function assumes a min of 2 light levels")

    # Define time vector
    dt = 1/1000
    tot_dur = np.sum(light_dur) + np.sum(ramp_dur)
    time = np.linspace(0, tot_dur, int(round(tot_dur/dt)))

    # Make empty dataframe
    df = pd.DataFrame(columns=['light_level','control_level'], index=time, dtype='float')    

    # Initial light_level values
    df.loc[df.index<=light_dur[0], 'light_level'] = light_level[0]

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

def run_program(df, dmx, aud_path, trig_video=True):
    """ 
    Transmits signal to control light intensity via Enttex DMX USB Pro.
    df         - Dataframe with values for control signal and desired light intensity
    dmx        - specifies the hardware address for the Enttex device
    aud_path   - Path to audio file to play for timecode
    trig_video - Whether to trigger the video via timecode audio
    """

    # Audio control described here:
    # https://stackoverflow.com/questions/57158779/how-to-stop-audio-with-playsound-module

    # Sets DMX channel 1 to max 255 (Channel 1 is the intensity)
    dmx.set_channel(1, 255)  

    # Timer starts
    start_time = time.time()
    curr_time  = 0
    end_time   = max(df.index)

    if trig_video:
        p = multiprocess.Process(target=playsound, args=(aud_path, ))
    
        print('Starting audio for timecode . . .')
        p.start()

    # Loop until time runs out
    while curr_time<end_time:          
    
        # Total time elapsed since the timer started
        curr_time = time.time() - start_time

        # Current control value (0 to 1)
        curr_control = np.interp(curr_time, df.index, df.control_level)

        # Sets DMX channel 1 in 8-bit value (Channel 1 is the intensity)
        dmx.set_channel(1, int(curr_control*255))  

        # Report status
        print("Time (s): " + str(round(curr_time,2)) + ", Writing to channel 1: " + str(round(curr_control,2)) )

        # Briefly pause the code to keep from overloading the hardware
        time.sleep(0.001)
        
    if trig_video:
        p.terminate()