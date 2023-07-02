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


def timecode_to_seconds(timecode):
    # Split the timecode into hours, minutes, seconds, and frames
    hours, minutes, seconds, frames = map(int, timecode.split(':'))

    # Calculate the total number of frames
    total_frames = frames + (seconds * 30) + (minutes * 60 * 30) + (hours * 60 * 60 * 30)

    # Convert frames to seconds (assuming 30 frames per second)
    total_seconds = total_frames / 30.0

    return total_seconds


def get_mask(im_mask_path):
    """Return mask image from path.
    Args:
        im_mask_path (str): Path to mask image.
    Returns:
        mask (np.array): Mask image.
        perim (np.array): Perimeter of mask.
        """

    # Read mask image
    mask = cv2.imread(im_mask_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the mask
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

    # mask = cv2.bitwise_not(mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it represents the dark region in the center)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the perimeter coordinates
    perim = np.squeeze(largest_contour)

    return mask, perim


def read_frame(cap, idx, im_mask=None, mask_perim=None, im_crop=False, outside_clr='white', color_mode='grayscale'):
    """Return frame at index idx from video capture object. Also enhance if desired.
        args:
            cap: Video capture object
            idx: Index of frame to read
            enhance_type: Type of contrast enhancement to apply
            im_mask: Mask image
            mask_perim: Needs to be provided if im_mask is given (generated by get_mask())
            im_crop: Whether to crop the image to the region of interest (requires mask and mask_perim)
            outside_clr: Color to fill outside of region of interest
            color_mode: Color mode ('RGB' or 'grayscale')
        returns:
            frame: Frame at index idx"""

    # Check that mask_perim is defined, if im_mask is given
    if im_mask is not None:
        if mask_perim is None:
            raise ValueError("mask_perim must be defined if im_mask is given.")
        
    if im_crop is True:
        if im_mask is None:
            raise ValueError("im_mask must be defined if im_crop is True.")
        
    # Read frame at index idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    _, frame = cap.read()

    if color_mode=='grayscale':
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if frame is  None:
        print(f"Invalid frame at index {idx}")
    
    # Mask frame, if mask image given
    if im_mask is not None:

        # Apply mask to frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=im_mask)

    # Apply color outside of image
    if outside_clr=='common':
        if color_mode=='grayscale':
            # Calculate the mode of the masked frame
            mode = int(np.median(masked_frame[masked_frame > 0]))

            # Set the area outside the ellipse to the mode
            masked_frame[np.where(im_mask == 0)] = mode
        else:
            # Calculate the mode individually for each color channel within the masked frame
            channel_modes = [int(np.median(masked_frame[:, :, i][masked_frame[:, :, i] > 0])) for i in range(3)]

            # Create the most common RGB color from the channel modes
            most_common_color = tuple(channel_modes)

            # Set the area outside the ellipse to the most common color
            masked_frame[np.where(im_mask == 0)] = most_common_color

    elif outside_clr=='gray':
        if color_mode=='grayscale':
            masked_frame[np.where(im_mask == 0)] = 128
        else:
            # Set the area outside the ellipse to gray
            masked_frame[np.where(im_mask == 0)] = (128, 128, 128)

    elif outside_clr=='white':
        if color_mode=='grayscale':
            masked_frame[np.where(im_mask == 0)] = 255
        else:
            # Set the area outside the ellipse to gray
            masked_frame[np.where(im_mask == 0)] = (255, 255, 255)

    elif outside_clr=='black':
        if color_mode=='grayscale':
            masked_frame[np.where(im_mask == 0)] = 0
        else:
            # Set the area outside the ellipse to gray
            masked_frame[np.where(im_mask == 0)] = (0, 0, 0)

    # Replace the frame with the masked frame
    frame = masked_frame

    # Crop image to the perimeter of the mask
    if im_crop:
        frame = frame[mask_perim[:, 1].min():mask_perim[:, 1].max(), mask_perim[:, 0].min():mask_perim[:, 0].max()]

    return frame


def make_max_mean_image(cat_curr, sch, vid_path, max_num_frames, im_mask=None, mask_perim=None, im_crop=False, vid_ext_raw='MOV'):
    """Create a mean image from a video capture object. Selects frames from movies where the light is set at maximum intensity for the batch.
        args:
            cat_cur: Subset of experiment_log to be included in mean image
            sch: Schedule of experiments
            vid_path: Path to video
            max_num_frames: Maximum number of frames to include in mean image
            im_mask: Mask image
            mask_perim: Needs to be provided if im_mask is given (generated by get_mask())
            im_crop: Whether to crop the image to the region of interest (requires mask and mask_perim)
            vid_ext_raw: File extension of raw video
        returns:
            mean_image: Mean image"""

    # Find the maximum value among light_start, light_end, and light_return, and light_btwn in sch 
    max_light = max(sch['light_start'].max(), sch['light_end'].max(), sch['light_return'].max())

    # Find values of light_start that equal max_light
    light_start_max  = sch['light_start'] == max_light
    light_end_max    = sch['light_end'] == max_light
    light_return_max = sch['light_return'] == max_light

    # TODO: Add timecode_start to starting time

    # Calculate the starting and ending times for each light_start, light_end, and light_return
    start_starttime  = np.zeros(sch.shape[0])
    start_endtime    = start_starttime + 60*sch['start_dur_min'].values
    end_starttime    = start_endtime + sch['ramp_dur_sec'].values
    end_endtime      = end_starttime + 60*sch['end_dur_min'].values
    return_starttime = end_endtime + sch['ramp2_dur_sec'].values
    return_endtime   = return_starttime + 60*sch['return_dur_min'].values

    # make new a pandas dataframe that lists the video_filename from cat_curr, but only for the rows for light_start_max or light_end_max or light_return_max
    # and only for the columns 'video_filename', 'light_start', 'light_end', 'light_return', 'light_btwn'
    data = {'vid_files': cat_curr.loc[:, 'video_filename'].values,
            'start_startime':np.nan,
            'start_endtime':np.nan,
            'end_starttime':np.nan,
            'end_endtime':np.nan,
            'return_starttime':np.nan,
            'return_endtime':np.nan}
    df = pd.DataFrame(data=data)

    # Add the start and end times to the dataframe for times where light is at max
    df.loc[light_start_max, 'start_starttime']    = start_starttime[light_start_max]
    df.loc[light_start_max, 'start_endtime']     = start_endtime[light_start_max]
    df.loc[light_end_max, 'end_starttime']       = end_starttime[light_end_max]
    df.loc[light_end_max, 'end_endtime']         = end_endtime[light_end_max]
    df.loc[light_return_max, 'return_starttime'] = return_starttime[light_return_max]
    df.loc[light_return_max, 'return_endtime']   = return_endtime[light_return_max]

    # Remove rows in df where idx is False
    df = df[light_start_max | light_end_max | light_return_max]

    # Get the maximum number of frames per video
    frames_per_vid = int(max_num_frames/df.shape[0])

    # Make loop through the video_filename in df
    for idx, vid_file in enumerate(df['vid_files'].values):

        full_path = os.path.join(vid_path, vid_file + '.' + vid_ext_raw)
        vid = cv2.VideoCapture(full_path)
        fps = vid.get(cv2.CAP_PROP_FPS)

        # Get the start and end times for the current video that are not nan values
        start_start  = df.loc[idx, 'start_starttime'] 
        start_end    = df.loc[idx, 'start_endtime']
        end_start    = df.loc[idx, 'end_starttime']
        end_end      = df.loc[idx, 'end_endtime']
        return_start = df.loc[idx, 'return_starttime'] 
        return_end   = df.loc[idx, 'return_endtime'] 

        # Collect frame numbers for snippets at max light intensity, for current video
        frame_nums = np.array([])
        if not np.isnan(start_start):
            frame_nums = np.append(frame_nums, np.arange(int(start_start*fps)+1, int(start_end*fps)-1, frames_per_vid))

        if not np.isnan(end_start):
            frame_nums = np.append(frame_nums, np.arange(int(end_start*fps)+1, int(end_endframe*fps)-1, frames_per_vid))

        if not np.isnan(return_start):
            frame_nums = np.append(frame_nums, np.arange(int(return_start*fps)+1, int(return_end*fps)-1, frames_per_vid))

        # Randomly select frames_per_vid frames from the frames array
        frame_nums = np.random.choice(frame_nums, frames_per_vid, replace=False)

        # Loop through the frame_nums array
        for frame_num in frame_nums:
            
            # Import the current frame
            frame = read_frame(vid, frame_num, im_mask, mask_perim, im_crop=im_crop)

            # if mean_image does not exist, create it
            if 'mean_image' not in locals():
                mean_image = frame.astype('float')
                num_frames = 1
            # otherwise, add the current frame to mean_image
            else:
                mean_image += frame.astype('float')
                num_frames += 1

        vid.release()

        # Update status
        print('Finished video {} of {}'.format(idx+1, df.shape[0]))

    # Divide mean_image by num_frames to get the mean
    mean_image = mean_image/num_frames

    # Convert mean image to uint8
    mean_image = np.round(mean_image).astype(np.uint8)

    return mean_image