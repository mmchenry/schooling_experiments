"""
    Series of functions for manipulating and interacting with video. Requires installing ffmpeg and opencv.
"""

# Imports
import cv2 as cv  # openCV for interacting with video
import os
import sys
import csv
import pathlib
import glob
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt


def vid_from_seq(imPath, vidPath=None, frStart=None, frEnd=None, fps=30, imQuality=0.75, prefix='DSC',
                 nDigits=5, inSuffix='JPG', outSuffix='mp4', roi=None, vertPix=None, 
                 vMode=False):
    """Creates a movie from an image sequence.
       imPath (str)      - Path to directory holding image file.
       vidPath (str)     - Path to output video file. Defaults to imPath.
       frStart (int)     - Frame number to begin working with. Will use first frame, if not specified.
       frEnd (int)       - Frame number to end with. Will use last frame, if not specified.
       fps (float)       - Frame rate (frames per sec) of input images. Will use same for output.
       imQuality (float) - image quality (0 - 1).
       prefix (str)      - Text that the image filenames begin with.
       nDigits (int)     - Number of digits in image filenames.
       inSuffix (str)    - Suffix of image file names.
       outSuffix (str)   - Suffix for movie file.
       roi (int)         - Region-of-interest coordinates (in pixels): [x y w h]
       vertPix (int)     - Size of video frames in vertical pixels 
       vMode (bool)      - Verbose mode, shows more output from ffmpeg
    """

    # set downSample
    if vertPix is None:
        downSample = False
    else:
        downSample = True 

    # Default to pwd, if no out path
    if vidPath is None:
        vidPath = imPath + os.path.sep + 'output_video' + '.' + outSuffix

    # Quality value, on the 51-point scale used by ffmpeg
    qVal = 51 * (1 - imQuality)

    # Check for path
    if not os.path.isdir(imPath):
        raise ValueError('Image path not found: ' + imPath)  

    # Listing of image files
    imFileList = glob.glob(imPath + os.path.sep + prefix + '*.' + inSuffix)

    # Check imFileList
    if len(imFileList)==0:
        raise ValueError('No images with ' + inSuffix + ' extension found in ' + imPath) 
   
    # Find start and end frames, if none given
    if (frStart is None) or (frEnd is None):
        frNums = []
        # Loop thru filenames and record numbers
        for cFile in imFileList:
            frNums.append(int(cFile[-(len(inSuffix)+nDigits+1):-(len(inSuffix)+1)]))

        # Check frame interval numbering
        if min(np.diff(np.sort(frNums))) != max(np.diff(np.sort(frNums))):
            raise ValueError('Interval between frame numbers not equal among all files')
        
        # Define range of frame numbers
        frStart = min(frNums)
        frEnd   = max(frNums)

    # Total number of frames
    nFrames = frEnd - frStart

    # Round roi coords down to an even number of pixels
    if roi is not None:
        roi[0] = int(2 * np.floor(roi[0] / 2))
        roi[1] = int(2 * np.floor(roi[1] / 2))
        roi[2] = int(2 * np.floor(roi[2] / 2))
        roi[3] = int(2 * np.floor(roi[3] / 2))

    # Figure horiz dimension
    if downSample:
        # If no roi provided
        if roi is None:
            imFiles = glob.glob(imPath + os.path.sep + prefix + '*' + inSuffix)
            im = cv.imread(imFiles[0])
            AR = im.shape[1]/im.shape[0]
        
        # Using roi
        else:
            # Round roi coords down to an even number of pixels
            roi[2] = int(2*np.floor(roi[2]/2))
            roi[3] = int(2*np.floor(roi[3]/2))
            AR     = roi[2]/roi[3]

        # Find horizontal dimension, rounding down to an even number
        horzPix = int(2*np.floor(vertPix * AR/2))

    # Match output with input frame rate
    fpsOut = fps

    # Start building the ffmpeg command
    command = f"ffmpeg -framerate {fps} -start_number {frStart} -i {imPath} {os.path.sep} {prefix}%0{nDigits}d.{inSuffix} -vframes {nFrames} "
    
    # Commands for an image sequence. Note that "-loglevel quiet" makes ffmpeg less verbose. Remove that for troubleshooting
    if not vMode:
        command += f"-loglevel quiet "
    
    # Specify compression
    command += f"-y -vcodec libx264 -pix_fmt yuv420p -an -r {fpsOut} -crf {qVal} "

    # Add downsampling and cropping commands
    if roi is not None:
        if downSample:
            # command += f"\"crop= {r[2]}:{r[3]}:{r[0]}:{r[1]}\" "
            command += f"-vf \"crop= {roi[2]}:{roi[3]}:{roi[0]}:{roi[1]}, scale={horzPix}:{vertPix}\" "
        else:
            command += f"-vf \"crop= {roi[2]}:{roi[3]}:{roi[0]}:{roi[1]}\" "
    elif downSample:
        command += f"-vf \"scale={horzPix}:{vertPix}\" "

    # Specify output file
    command += f"-timecode 00:00:00:00 '{vidPath}'"

    # Report attempt
    print('    Reading images from: ' + imPath)
    print('    Making output movie file: ' + vidPath)

    # Excute ffmpeg
    os.system(command)

    # Wrap up
    print('    Completed writing ' + str(nFrames) + ' frames')

 
def vid_convert(vInPath, vOutPath, imQuality=1, roi=None, vertPix=None, 
                vMode=True, maskpath=None, para_mode=False, echo=True):
    """Converts a video file, perhaps with cropping and downsampling.
       vInPath (str)     - Path to input video file.
       vOutPath (str)    - Path to output video file. Defaults to same as vInPath.
       fps (float)       - Frame rate (frames per sec) of input images. Will use same for output.
       imQuality (float) - image quality (0 - 1).
       roi (int)         - Region-of-interest coordinates (in pixels): [x y w h]
       vertPix (int)     - Size of video frames in vertical pixels 
       vMode (bool)      - Verbose mode, shows more output from ffmpeg
       maskpath          - Path to PNG mask file (transparent pixels are for visible parts of video)
       para_mode         - Mode for parallel processing, where the unix command is not executed

       Note: if you are masking a video, you cannot downsample or crop it
    """

    # Check for crop, downsampling, and masking
    if (maskpath is not None) and (roi is not None):
        raise ValueError('You cannot both crop and mask the video -- pick one.')
    elif (maskpath is not None) and (vertPix is not None):
        raise ValueError('You cannot both downsample and mask the video -- pick one.')

    # Check extension of mask
    if maskpath is not None:
        pathparts = os.path.splitext(maskpath)
        if pathparts[1] != '.png':
            raise ValueError('Mask file needs to be PNG format, with some transparent pixels')

    # overwrite existing file
    overWrite = True

    # Remove audio
    noAudio = True

    # Check for path
    if not os.path.isfile(vInPath):
        raise ValueError('Movie not found at given path: ' + vInPath) 

    # Quality value, on the 51-point scale used by ffmpeg
    qVal = 51 * (1 - imQuality)

    # Round roi coords down to an even number of pixels
    if roi is not None:
        roi[0] = int(2 * np.floor(roi[0] / 2))
        roi[1] = int(2 * np.floor(roi[1] / 2))
        roi[2] = int(2 * np.floor(roi[2] / 2))
        roi[3] = int(2 * np.floor(roi[3] / 2))

    # Figure horiz dimension
    if vertPix is not None:
        # If no roi provided
        if roi is None:
            im = get_frame(vInPath)
            AR = im.shape[1]/im.shape[0]
        
        # Using roi
        else:
            AR = roi[2]/roi[3]

        # Find horizontal dimension, rounding down to an even number
        horzPix = int(2*np.floor(vertPix * AR/2))

    # Start building the ffmpeg command
    command = f"ffmpeg -i {vInPath} "

    # If there is a mask
    if (maskpath is not None):
        command += f"-i {maskpath} -filter_complex \"[0:v][1:v] overlay=0:0\" "

    # Whether to overwrite existing file
    if overWrite:
        command += "-y "

    # Whether to remove audio
    if noAudio:
        command += "-an "
    
    # "-loglevel quiet" makes ffmpeg less verbose. Remove that for troubleshooting
    if not vMode:
        command += f"-loglevel quiet "
    
    # Specify compression
    command += f"-vcodec libx264 -pix_fmt yuv420p -an -crf {qVal} "

    # Add downsampling and cropping commands
    if roi is not None:
        if vertPix is not None:
            # command += f"\"crop= {r[2]}:{r[3]}:{r[0]}:{r[1]}\" "
            command += f"-vf \"crop= {roi[2]}:{roi[3]}:{roi[0]}:{roi[1]}, scale={horzPix}:{vertPix}\" "
        else:
            command += f"-vf \"crop= {roi[2]}:{roi[3]}:{roi[0]}:{roi[1]}\" "
    elif vertPix is not None:
        command += f"-vf \"scale={horzPix}:{vertPix}\" "


    # Specify output file
    command += f"-timecode 00:00:00:00 '{vOutPath}'"

    # # Report attempt
    if echo:
        print('    Making output movie file: ' + vOutPath)

    if not para_mode:
        # Excute ffmpeg
        os.system(command)

    return command


def get_frame(vid_path, fr_num=1):
    """ Reads a single frame from a video file.

    vid_path:   Full path to the video file
    fr_num:     Frame number to be extracted
    """

    # Check for file existance
    if not os.path.isfile(vid_path):
        raise Exception("Video file does not exist at: " + vid_path)

    # Define video object &  video frame
    vid = cv.VideoCapture(vid_path)

    # Video duration (in frames)
    frame_count = int(vid.get(cv.CAP_PROP_FRAME_COUNT))

    if fr_num > frame_count:
        raise Exception('Frame number requested exceeds video duration')
    else:
        vid.set(cv.CAP_PROP_POS_FRAMES, fr_num)
        _, frame = vid.read()

        return frame

def find_roi(in_path, fr_num=1, show_crosshair=True, from_center=False):
    """Reads frame of video and prompts to interactively select a roi.
        in_path (str)           - Can be a path to a movie or image
        fr_num (int)            - Frame number of movie used to select the roi
        show_crosshair (bool)   - Whether to show the cross hairs
        from_center (bool)      - Whether to start drawing the roi from its own center
    """

    # Get extension to movie/image file
    fName, ext = os.path.splitext(in_path)

    # Chec if it's a movie
    if ext=='.mov' or ext=='.mp4' or ext=='.avi':
        isMovie = True
    else:
        isMovie = False

    if isMovie:
        # Define video object &  video frame
        vid = cv.VideoCapture(in_path)

        # Get frame and select roi
        im0 = get_frame(in_path, fr_num)
    else:
        # Get frame and select roi
        im0 = cv.imread(in_path)

    # Create named window
    cv.namedWindow("ROI_Select", cv.WINDOW_NORMAL)
    cv.startWindowThread()

    # Select ROI
    r = cv.selectROI("ROI_Select", im0, show_crosshair, from_center)
    cv.waitKey()
    cv.destroyWindow("ROI_Select")

    # close window 
    cv.waitKey(1)
    cv.destroyAllWindows()

    # Release video capture
    if isMovie:
        vid.release()

    return r


def find_coords(vid_path, poly_overlay=False, num_pts=inf, fr_num=1):
    """Reads frame of video and prompts to interactively select coordinates
    
    vid_path: full path to video file
    poly_overlay: overlays a polygon area, if set to True
    num_pts: Number of points to collect per frame
    fr_num: Frame number of video file for the coordinate acquisition
    
    """

    # get access to a couple of global variables we'll need
    global coords, drawing

    # Initialize container for coordinates
    coords = []

    # Define video object &  video frame
    vid = cv.VideoCapture(vid_path)

    # Get frame
    im0 = get_frame(vid_path, fr_num)

    # Create named window
    cv.namedWindow("Coord_Select", cv.WINDOW_GUI_EXPANDED)
    cv.startWindowThread()

    # Select coordinates
    cv.setMouseCallback('Coord_Select', click_coords, im0)

    # Loop for collecting coordinates/keyboard inputs
    while 1==1:
        cv.imshow("Coord_Select",im0)
        k = cv.waitKey(20) & 0xFF
        if k == 27:
            break
        elif len(coords)==num_pts:
            break

    # Release video capture and close window
    vid.release()
    cv.waitKey(1)
    cv.destroyAllWindows()

    if poly_overlay:
        # Overlay points on video
        cv.imshow("Coord_Display", cv.WINDOW_NORMAL)
        cv.startWindowThread()

        # Construct polgon
        polygon = [np.int32(coords)]
        im0 = cv.polylines(im0, polygon, False, (0, 255, 0), thickness=2)

        # Interactive mode
        while True:
            cv.imshow("Coord_Display",im0)
            k = cv.waitKey(20) & 0xFF
            if k == 27:
                break

        # Release video capture and close window
        vid.release()
        cv.waitKey(1)
        cv.destroyAllWindows()
    

    return coords


def click_coords(event, x, y, flag, image):
    """
    Callback function, called by OpenCV when the user interacts
    with the window using the mouse. This function will be called
    repeatedly as the user interacts.
    """
    # get access to a couple of global variables we'll need
    global coords, drawing

    if event == cv.EVENT_LBUTTONDOWN:
        # user has clicked the mouse's left button
        drawing = True

        # Marker at the selected coordinates
        cv.circle(image,(x,y),3,(0,255,0),-1)

        # Add coordinates
        coords.append((x, y))


def get_background(vid_path, out_path, max_frames):
    """Computes background of video and outputs as png
    
    vid_path: Full path to video file
    out_path: Full path to output png file
    max_frames: Max number of frames for calculating the average background 
    """

    # Create video capture object, check if video exists
    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        sys.exit(
            'Video cannot be read! Please check vid_path to ensure it is correctly pointing to the video file')

    # Create background object
    bgmodel = cv.createBackgroundSubtractorMOG2()

    # Get first frame for setting up output video
    ret, frame_init = cap.read()

    # Resize dimensions (for image preview only)
    resize_dim = (int(frame_init.shape[1] // 3), int(frame_init.shape[0] // 3))

    # Video duration (in frames)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Set max index for background model
    if frame_count >= max_frames:
        ind_max = max_frames
    else:
        ind_max = frame_count

    # Create window for viewing current output frame
    cv.namedWindow("bg Model", cv.WINDOW_NORMAL)

    # Text and parameters for frame number overlay
    font = cv.FONT_HERSHEY_SIMPLEX
    text_pos = (400, 100)
    font_scale = 2
    font_color = (155, 155, 155)
    font_thickness = 2

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert current frame to grayscale
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Extract current frame number
        frame_curr = cap.get(1)

        if ret:
            # Apply background model, returns binary foreground image
            fgmask = bgmodel.apply(frame)

            # Get background image
            bgImage = bgmodel.getBackgroundImage()

            # Copy background image and add text for showing progress
            bg_copy = bgImage.copy()
            cv.putText(bg_copy, 'Frame: ' + str(frame_curr),
                       text_pos,
                       font,
                       font_scale,
                       font_color,
                       font_thickness, cv.LINE_AA)

            # Show background model progress
            cv.imshow("bg Model", cv.resize(bg_copy, resize_dim))
            cv.waitKey(20)

            # Close window and break loop with 'esc' key
            k = cv.waitKey(20) & 0xff
            if k == 27:
                break

        # Save background image and Break while loop after max frames
        if frame_curr >= ind_max:
            # Write background image
            cv.imwrite(out_path, bgImage)
            break

    print('Background image complete')

    # When everything done, release the capture
    cap.release()
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)
    cv.waitKey(1)

    return bgImage


def bg_subtract(vid_path, out_path, roi):
    """Perform background subtraction and image smoothing to video"""

    # Open video, check if it exists
    cap = cv.VideoCapture(vid_path)

    if not cap.isOpened():
        sys.exit(
            'Video cannot be read! Please check vid_path to ensure it is correctly pointing to the video file')

    # Set codec for output video
    # codec = 'mp4v'
    codec = 'MJPG'

    # Open background image
    bgImg = cv.imread(out_path + '-bgImg.png')

    # Check if background image was loaded
    if bgImg is None:
        sys.exit("Could not read the image. Pathname incorrect OR needs to run getbackground")

    # Convert background image to grayscale
    bg_gray = cv.cvtColor(bgImg.copy(), cv.COLOR_BGR2GRAY)

    x1 = roi[0]
    y1 = roi[1]
    x2 = roi[2]
    y2 = roi[3]

    out_vid_path = out_path + '-bgSub.mp4'

    # Output frame size set by mask radius, which will be used for cropping video
    output_framesize = (int(y2), int(x2))

    # Video writer class to output video with pre-processing
    fourcc = cv.VideoWriter_fourcc(*codec)

    # Create video output object
    out = cv.VideoWriter(filename=out_vid_path, fourcc=fourcc, fps=30.0, frameSize=output_framesize, isColor=False)

    # Create a CLAHE object for histogram equalization
    clahe = cv.createCLAHE(clipLimit=6.0, tileGridSize=(61, 61))

    # Create window for viewing current output frame
    cv.namedWindow('frame_curr', cv.WINDOW_NORMAL)

    # Initialize variable to break while loop when last frame is achieved
    last = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Extract current frame number
        this_frame = cap.get(1)

        if ret:
            # Convert current frame to grayscale
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Take complement of background
            bg_inv = cv.bitwise_not(bg_gray)

            # Background subtraction (by adding inverse of background)
            frame_sub = cv.add(frame, bg_inv)

            # Apply histogram equalization to background subtracted image
            frame_adjust = clahe.apply(frame_sub)

            # Apply smoothing filter
            frame_adjust = cv.bilateralFilter(frame_adjust, 5, 40, 40)

            # Crop image
            frame_crop = frame_adjust[int(y1):int(y1 + y2), int(x1):int(x1 + x2)]

            # Write current processed frame to output object
            out.write(frame_crop)

            # Display output image (bgSubtract + processed + cropped)
            cv.imshow('frame_curr', frame_crop)
            if cv.waitKey(1) == 27:
                break

        if last >= this_frame:
            break

        last = this_frame

    print("Background subtraction complete")

    # When everything done, release the capture
    cap.release()
    out.release()
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)
    cv.waitKey(1)


