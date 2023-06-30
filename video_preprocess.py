"""
Functions for preprocessing video files for analysis by TRex and TGrabs.
"""

import os
import cv2
import numpy as np
import pandas as pd
import subprocess


def find_schedule_matches(csv_dir, match_dir):
    """Find matching video files in the match directory based on the first 10 characters of the filename
    Args:
        csv_dir (str): Path to the directory containing the CSV files.
        match_dir (str): Path to the directory containing the matching files.
    Returns:
        matching_files (list): List of filenames that match.
        other_files (list): List of filenames that do not match.
    """

    # Get filenames of CSV files without extensions
    csv_files = [os.path.splitext(file)[0] for file in os.listdir(csv_dir) if file.lower().endswith('.csv')]

    # Check for matches in the match directory based on the first 10 characters
    matching_files = []
    other_files = []
    for file in csv_files:
        if file[:10] in os.listdir(match_dir):
            matching_files.append(file)
        else:
            other_files.append(file)

    return matching_files, other_files


def check_video_duration(vid_dir, sch, cat, vid_ext='MOV', thresh_time=3.0):
    """Check the duration of each video file in a directory and compare that to the expected duration from sch dataframe.
    Args:
        vid_dir (str): Path to the directory containing the video files.
        sch (pandas.DataFrame): Pandas dataframe containing the recording sch for current schedule.
        cat (pandas.DataFrame): Pandas dataframe containing the recording log.
        vid_ext (str): Video file extension.
        thresh_time (float): Time (in sec) to allow for video to be shorter than expected.
    Returns:
        vid_files (list): List of filenames in current schedule.
    """
  
    # Get filenames of video files without extensions
    vid_files = get_matching_video_filenames(cat, vid_dir, vid_ext=vid_ext)

    # Loop through each video file
    for file in vid_files:

        # Get the duration of the video file
        vid_path = os.path.join(vid_dir, file)
        vid = cv2.VideoCapture(vid_path)
        vid_duration = vid.get(cv2.CAP_PROP_FRAME_COUNT) / vid.get(cv2.CAP_PROP_FPS)
        vid.release()

        # Get the trial number (trial_num) from cat that matches the current video file
        curr_trialnum = cat.loc[cat['video_filename'] == os.path.splitext(file)[0]].trial_num

        # Define sch_row as the row of values where trial_num==curr_trialnum
        sch_row = sch.loc[sch['trial_num'] == curr_trialnum.values[0]]

        # Calculate the duration of the recording
        if sch_row['ramp2_dur_sec'].values.size == 0:
            sch_duration = 60*sch_row['start_dur_min'].values[0] + sch_row['ramp_dur_sec'].values[0] + 60*sch_row['end_dur_min'].values[0] 
        else:
            sch_duration = 60*sch_row['start_dur_min'].values[0] + sch_row['ramp_dur_sec'].values[0] + 60*sch_row['end_dur_min'].values[0] + sch_row['ramp2_dur_sec'].values[0] + 60*sch_row['return_dur_min'].values[0]

        # Check for various problems
        if vid_duration > sch_duration:
            print("Video duration is GREATER than the duration in the experiment_log: " + file)
            print('   Video duration: ' + "{:.1f}".format(vid_duration))
            print('   sch duration: ' + "{:.1f}".format(sch_duration))


        elif (sch_duration - vid_duration)>thresh_time:
            print("Video duration is LESS THAN the duration in experiment_log by more than " + "{:.1f}".format(thresh_time) + " s: " + file)
            print('   Video duration: ' + "{:.1f}".format(vid_duration))
            print('   sch duration: ' + "{:.1f}".format(sch_duration))  

    return vid_files

def get_matching_video_filenames(cat, directory, vid_ext='MOV'):
    """Get the filenames of the video files in a directory that match the video filenames in the cat dataframe.
    Args:
        cat (pandas.DataFrame): Pandas dataframe containing the recording log.
        directory (str): Path to the directory containing the video files.
        vid_ext (str): Video file extension.
    Returns:
        matching_video_filenames (list): List of filenames that match.
        """
    video_filenames = cat['video_filename'].tolist()
    matching_video_filenames = []

    for filename in os.listdir(directory):
        if filename.endswith('.' + vid_ext) and os.path.splitext(filename)[0] in video_filenames:
            # matching_video_filenames.append(os.path.splitext(filename)[0])
            matching_video_filenames.append(filename)

    return matching_video_filenames


def add_start_timecodes(vid_files, vid_path, cat):
    for file in vid_files:
        filename = os.path.splitext(file)[0]  # Extract filename without extension
        
        # Find matching row in cat DataFrame based on video_filename
        match_row = cat[cat['video_filename'] == filename]
        
        if not match_row.empty:
            video_path = os.path.join(vid_path, file)
            cap = cv2.VideoCapture(video_path)
            
            # Use ffprobe to get the video duration
            ffprobe_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 '{video_path}'"
            result = subprocess.run(ffprobe_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                duration = float(result.stdout)
                # fps = match_row['fps'].iloc[0]  # Extract the frame rate from the DataFrame
                fps = cap.get(cv2.CAP_PROP_FPS)
                start_time = "00:00:00:00"  # Default start timecode
                
                if fps > 0:
                    frames = int(fps * duration)
                    start_time = f"{frames // 3600:02d}:{frames // 60 % 60:02d}:{frames % 60:02d}:00"
                
                # Update timecode_start in matching row of cat DataFrame
                cat.loc[match_row.index, 'timecode_start'] = start_time
    
    return cat
