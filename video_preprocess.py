"""
Functions for preprocessing video files for analysis by TRex and TGrabs.
"""

import os
import cv2
import numpy as np
import pandas as pd
import subprocess
import gui_functions as gf
import time


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

    if frame is None:
        raise ValueError(f"Invalid frame at index {idx}")

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


def threshold_diff_image(im, im_mean, threshold):
    """Generates binary image based on difference from mean image
    Args:
        im (np.array): Image to threshold.
        im_mean (np.array): Mean image.
        threshold (int): Threshold value.
    Returns:
        im_thresh (np.array): Thresholded image.
        """
    # Compute absolute difference between im and im_mean
    im_diff = cv2.absdiff(im, im_mean)

    # Apply a threshold 
    _, im_thresh = cv2.threshold(im_diff, threshold, 255, cv2.THRESH_BINARY)

    return im_thresh


def filter_blobs(im, im_mean, threshold, min_area, max_area, max_aspect_ratio=0.1, white_blobs=False):
    """Filter blobs based on area and aspect ratio.
    Args:
        im (np.array): Image to threshold.
        im_mean (np.array): Mean image.
        threshold (int): Threshold value.
        min_area (int): Minimum area.
        max_area (int): Maximum area.
        max_aspect_ratio (float): Maximum aspect ratio (width/height) allowed for blobs.
        white_blobs (bool): If True, filter white blobs. If False, the images within the blobs are visible.
    Returns:
        filtered_im (np.array): Filtered image.
    """

    # Convert image to binary
    im_thresh = threshold_diff_image(im, im_mean, threshold)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter individual blobs based on area and aspect ratio
    filtered_contours = []
    for contour in contours:
        # Calculate the total area of white pixels for each contour
        mask = np.zeros_like(im_thresh)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        total_area = np.sum(im_thresh[mask != 0] > 0)

        # Calculate the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the aspect ratio
        aspect_ratio = float(w) / h

        # Filter based on area and aspect ratio
        if min_area <= total_area <= max_area and aspect_ratio >= max_aspect_ratio:
            filtered_contours.append(contour)

    # Create a binary mask for the filtered contours
    mask = np.zeros_like(im_thresh)
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original image
    if not white_blobs:
        filtered_im = cv2.bitwise_and(im, im, mask=mask)
    else:
        # make the blobs white in the filtered image
        filtered_im = np.zeros_like(im)
        filtered_im[mask != 0] = 255

    return filtered_im



def fill_and_smooth(image, num_iterations=3, kernel_size=3):
    """Fill holes in the image and smooth the edges.
    Args:
        image (np.array): Image to fill and smooth.
        num_iterations (int): Number of dilation and erosion cycles.
        kernel_size (int): Size of the kernel for dilation and erosion.
    Returns:
        smoothed (np.array): Smoothed image.
    """

    # Find contours of white blobs
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for filling the blobs
    mask = np.zeros_like(image)

    # Fill each contour with white color
    for contour in contours:
        cv2.drawContours(mask, [contour], 0, 255, -1)

    # Perform cycles of dilation and erosion to smooth the blobs
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    smoothed = mask.copy()

    for _ in range(num_iterations):
        dilated = cv2.dilate(smoothed, kernel, iterations=1)
        smoothed = cv2.erode(dilated, kernel, iterations=1)
    
    # Dilate the smoothed image one more time
    dilated = cv2.dilate(smoothed, kernel, iterations=1)

    return smoothed


def make_binary_movie(vid_path_in, vid_path_out, mean_image, threshold, min_area, max_area, 
                      im_mask=None, mask_perim=None, im_crop=True, status_txt=None, thresh_tol=0.05, echo=False):
    """Make a binary movie from a video.
    Args:
        vid_path_in (str): Path to input video.
        vid_path_out (str): Path to output video.
        mean_image (np.array): Mean image.
        threshold (int): Threshold value.
        min_area (int): Minimum area.
        max_area (int): Maximum area.
        im_mask (np.array): Image mask.
        mask_perim (int): Mask perimeter.
        im_crop (bool): If True, crop the image.
        thresh_tol (float): Threshold tolerance. Determines what proportion of blob area to trigger change in threshold.
    Returns:
        None
    """

    # Check that mask_perim is defined, if im_mask is given
    if im_mask is not None:
        if mask_perim is None:
            raise ValueError("mask_perim must be defined if im_mask is given.")
        
    if im_crop is True:
        if im_mask is None:
            raise ValueError("im_mask must be defined if im_crop is True.")

    # Define the codec to use for video encoding
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the video codec (e.g., 'mp4v' for MP4)

    # Set up input video and properties
    vid_in  = cv2.VideoCapture(vid_path_in)
    num_frames = int(vid_in.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame, find initial total area of blobs
    im = read_frame(vid_in, 0, im_mask=im_mask, mask_perim=mask_perim, im_crop=True, outside_clr='white', color_mode='grayscale')
    im_thresh = filter_blobs(im, mean_image, threshold, min_area, max_area, white_blobs=True)  
    im_thresh2 = fill_and_smooth(im_thresh)
    total_area0 = np.sum(im_thresh2 > 0)

    # Create video writer object for am mp4 video that otherwise has the same properties as input video
    vid_out = cv2.VideoWriter(vid_path_out, fourcc, int(vid_in.get(cv2.CAP_PROP_FPS)), (im.shape[1], im.shape[0]), isColor=False)

    # Start time for calculating elapsed time
    start_time = time.time()  

    # Smaller number of frames for testing
    # num_frames = 100

    # Create loop thru all frames
    for frame_num in range(num_frames):
    # for frame_num in range(30):

        # This mode dictates whether the threshold is being adjusted
        adjusting_threshold = True

        # Previous value for threshold
        threshold_prev = threshold

        # Number of threshold adjustments
        adjustments = 0

        # Read the current frame
        im = read_frame(vid_in, frame_num, im_mask=im_mask, mask_perim=mask_perim, im_crop=True, outside_clr='white', color_mode='grayscale')

        while adjusting_threshold:

            # Apply the filter_blobs function to the current frame
            im_thresh = filter_blobs(im, mean_image, threshold, min_area, max_area, white_blobs=True)  

            # Fill and smooth the blobs in im_thresh
            im_thresh2 = fill_and_smooth(im_thresh)

            # Calculate the total area of the blobs
            total_area = np.sum(im_thresh2 > 0)

            # Quit loop if the area has not changed much
            if (total_area/total_area0) < (1+thresh_tol) and \
                (total_area/total_area0) > (1-thresh_tol):
                adjusting_threshold = False

            # Quit loop if threshold has been adjusted more than once and returned to initial value
            elif (adjustments > 1) and (threshold == threshold_prev):
                adjusting_threshold = False

            # Increase threshold if area has increased much
            elif (total_area/total_area0) >= (1+thresh_tol):
                threshold = threshold + 1
                adjustments += 1
                if echo:
                    print('   ' + status_txt + ' Frame {} : Increasing threshold to {}'.format(frame_num+1, threshold))

            # Decrease threshold if area has decreased much
            elif (total_area/total_area0) <= (1-thresh_tol):
                threshold = threshold - 1
                adjustments += 1
                if echo:
                    print('   ' + status_txt + ' Frame {} : Decreasing threshold to {}'.format(frame_num+1, threshold))

        # Write the frame to the output video
        vid_out.write(im_thresh2)

        # Print progress, every 100 frames
        if frame_num % 100 == 0:
            elapsed_time = time.time() - start_time
            frames_processed = frame_num + 1
            frames_remaining = num_frames - frames_processed
            time_per_frame = elapsed_time / frames_processed
            time_remaining = frames_remaining * time_per_frame / 60

            print('   ' + status_txt + ': Finished frame {} of {}. Estimated time remaining: {:.1f} min'.format(
                frame_num+1, num_frames, time_remaining))
    
    # close video
    vid_in.release()
    vid_out.release()

