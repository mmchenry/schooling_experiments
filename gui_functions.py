import tkinter as tk
import screeninfo
import os
import video_preprocess as vp
import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import math
from shapely.geometry import Polygon


# The following is used to generate GUIs
def get_screen_resolution():
    """Get the screen resolution.
    Returns:
        width (int): Screen width.
        height (int): Screen height."""
    
    screen_info = screeninfo.get_monitors()
    width = screen_info[0].width
    height = screen_info[0].height
    return width, height


def window_position(root):
    """Calculate the position for the GUI window to appear centered.
    Args:
        root (tk.Tk): Tkinter root object.
    Returns:
        position_x (int): X position of the window.
        position_y (int): Y position of the window.
    """

    # Get screen resolution
    screen_width, screen_height = get_screen_resolution()

    # Get the window size
    window_width = root.winfo_reqwidth()
    window_height = root.winfo_reqheight()

    # Calculate the position for the GUI window to appear centered
    position_x = int((screen_width - window_width) / 2)
    position_y = int((screen_height - window_height) / 2)

    return position_x, position_y


def create_cv_window(win_name):
    """Creates a window with a size that is 80% of the screen resolution.
        args:
            win_name: Name of the window
    """

    # Create a window to display the image
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Get the screen resolution
    screen_width, screen_height = get_screen_resolution()

    # Calculate the desired window size
    window_width   = int(screen_width * 0.8)
    window_height  = int(screen_height * 0.8)

    # Set the window size
    cv2.resizeWindow(win_name, int(window_width/1.5), int(window_height/1.5))


def select_item(input_list, prompt_text, font_size=40):
    """Create a GUI to allow the user to select a number from the list.
    Args:
        input_list (list): List of items to select from.
        prompt_text (str): Text to display in the GUI.
        font_size (int): Font size for the labels and buttons.
    Returns:
        selected_number (int): Selected number.
    """

    root = tk.Tk()
    root.title(prompt_text)

    # Define the font for the labels and buttons
    font = ("Arial", 40)

    # Declare the selected_number variable
    selected_number = None

    # Function to handle number button clicks
    def on_number_click(number):
        nonlocal selected_number
        selected_number = number
        root.destroy()

    # Create a label for the instruction
    instruction_label = tk.Label(root, text=prompt_text, font=font)
    instruction_label.pack(pady=20)

    # Create a frame to hold the number buttons
    frame = tk.Frame(root)
    frame.pack(pady=10)

    # Generate random numbers
    # random_numbers = generate_random_numbers()

    # Create buttons for each number
    for l_item in input_list:
        button = tk.Button(frame, text=l_item, font=font, command=lambda num=l_item: on_number_click(num))
        button.pack(side=tk.LEFT, padx=10)

    x_pos, y_pos = window_position(root)

    # Set the position of the GUI window on the screen
    root.geometry(f"+{x_pos}+{y_pos}")

    # Start the GUI main loop
    root.mainloop()

    return selected_number


def prompt_number(title, prompt, font_size):
    """Create a GUI to prompt the user to enter a number.
    Args:
        title (str): Title for the GUI window.
        prompt (str): Prompt text for the GUI.
        font_size (int): Font size for the labels and buttons.
    Returns:
        num_returned (float): Number entered by the user.
    """
    # Create the GUI window
    root = tk.Tk()
    root.title(title)

    # Create a custom font with the desired size
    custom_font = ("Arial", font_size)

    # Create the prompt label with custom font style
    prompt_label = tk.Label(root, text=prompt, font=custom_font)
    prompt_label.pack(pady=10)

    # Create the entry widget
    entry = tk.Entry(root, font=custom_font)
    entry.pack(pady=5)

    # Create the OK button
    def get_value():
        value = entry.get()
        root.destroy()
        root.quit()
        entered_value.set(value)  # Store the value in the entered_value StringVar

    ok_button = tk.Button(root, text="OK", command=get_value, font=custom_font)
    ok_button.pack(pady=10)

    entered_value = tk.StringVar()  # Variable to store the entered value

    # Run the dialog window
    root.mainloop()

    # Convert the entered value to a float
    num_returned = float(entered_value.get())

    return num_returned  # Return the entered value


def select_polygon(frame):
    """Select a polygon from an image.
    Args:
        frame (np.array): Image to select polygon from.
    Returns:
        points (list): List of points in the polygon.
    """

    # Global variables
    points = []

    # Make a copy of the frame
    frame_start = frame.copy()

    def display_image_points(frame, points):
        # Draw the polygon on the empty image
        curve_points = draw_curves(points, resolution=100)

        # Redraw the entire polygon
        frame[:] = frame_start.copy()
        if (len(points) > 1) and (len(points) <= 3):
            cv2.polylines(frame, [np.array(points)], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            # Use cv.ploylines to draw the curve_points

        elif len(points) > 3:
            cv2.polylines(frame, [curve_points.astype(np.int32)], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        # Show the frame with the updated polygon
        cv2.imshow('Frame', frame)

    def draw_polygon(event, x, y, flags, param):
        """Draws a polygon on the image based on the user's mouse clicks.
        args:
            event: The type of mouse event
            x: The x-coordinate of the mouse click
            y: The y-coordinate of the mouse click
            flags: Any relevant flags
            param: Any extra parameters
        """
        nonlocal points

        if event == cv2.EVENT_LBUTTONDOWN:
            # Add the clicked point to the list
            points.append((x, y))

            # Update image
            display_image_points(frame, points)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove the last point if right-clicked
            if len(points) > 0:
                # Remove the point from the list
                points.pop()

                # Update image
                display_image_points(frame, points)

        # Draw red cross-hairs at the cursor position
        frame_crosshair = frame.copy()
        cv2.line(frame_crosshair, (x, 0), (x, frame_crosshair.shape[0]), (0, 0, 255), 1)  # Vertical line
        cv2.line(frame_crosshair, (0, y), (frame_crosshair.shape[1], y), (0, 0, 255), 1)  # Horizontal line
        cv2.imshow('Frame', frame_crosshair)

    # Create a window and set the mouse callback
    create_cv_window('Frame')
    cv2.setMouseCallback('Frame', draw_polygon)

    # Instructions
    print("Click to select points for the polygon. Press 'Enter' to finish. Right-click to delete the last point.")

    # Display the initial frame
    cv2.imshow('Frame', frame)

    # Keep processing frames until 'Enter' is pressed
    while True:
        # Exit the loop if 'Enter' key is pressed
        if cv2.waitKey(1) == 13:  # 13 is the ASCII code for 'Enter' key
            break

    # Destroy the window
    cv2.destroyAllWindows()

    frame = frame_start.copy()

    # Return the polygon points
    return points


def draw_curves(points, resolution=100):
    """Draws a smooth curve through a set of points.
    Args:
        points (list): List of points in the polygon.
        resolution (int): Resolution of the curve.
    Returns:
        curve_points (np.array): List of curve points.
    """
    # if len(points) < 4:
    #     # Handle case when number of points is less than 4
    #     return np.array(points)

    # Convert points to numpy array
    polygon_points = np.array(points)

    # Close the polygon by appending the first point to the end
    polygon_points = np.append(polygon_points, [polygon_points[0]], axis=0)

    # Fit a closed B-spline curve through the polygon points
    k = 3  # Degree of the B-spline curve
    if len(polygon_points) <= k:
        return np.array(points)  # Return the original points if not enough points for fitting the curve

    tck, u = splprep([polygon_points[:, 0], polygon_points[:, 1]], k=k, s=0, per=True)

    # Generate smooth points on the B-spline curve
    t_smooth = np.linspace(0, 1, resolution)
    x_smooth, y_smooth = splev(t_smooth, tck)

    # Combine x and y coordinates into a single array
    curve_points = np.column_stack((x_smooth, y_smooth))

    return curve_points


def generate_binary_image(frame, points, resolution=100):
    """Generate a binary image from a polygon.
    Args:
        frame (np.array): Image to generate binary image from.
        points (list): List of points in the polygon.
        resolution (int): Resolution of the curve.
    Returns:
        binary_image (np.array): Binary image.
    """

    # Create an empty binary image
    binary_image = np.zeros_like(frame)

    # Convert curve points to CV_32S datatype
    curve_points = draw_curves(points, resolution=int(resolution)).astype(np.int32)

    # Fill the polygon region with white color (255)
    cv2.fillPoly(binary_image, [curve_points], color=(255, 255, 255))

    # Convert the image to grayscale
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

    return binary_image


def create_mask_for_batch(vid_file, mask_file):
    """Create a mask for a batch of videos.
    Args:
        vid_files (list): List of video files.
        vid_path (str): Path to the videos.
        mask_path (str): Path to save the mask.
        cat_curr (str): Current category.
        path_mask (str): Path to save the mask.
        vid_ext (str): Video file extension.
        font_size (int): Font size for the text.
    Returns:
        None
    """

    # raise exception if video file does not exist
    if not os.path.isfile(vid_file):
        raise Exception('Video file does not exist: ' + vid_file)

    # Read first video frame of first video in cat_curr.video_filename
    vid = cv2.VideoCapture(vid_file)
    ret, im_start = vid.read()
    vid.release()

    # Convert im_start to 4 channels (BGR-A) by adding an alpha channel
    # im_start2 = cv2.cvtColor(im_start, cv2.COLOR_BGR2BGRA)

    # make copy of im_start
    im_start2 = im_start.copy()

    # Call the select_polygon() function
    points = select_polygon(im_start)

    # End function if no points are selected
    if len(points) == 0:
        print(' ')
        print('No points selected. Exiting function.')
        return

    # get centroid of points polygon
    point_polygon = Polygon(points)
    polygon_centroid = point_polygon.centroid
    centroid_x = polygon_centroid.x
    centroid_y = polygon_centroid.y
    centroid = [int(centroid_x),int(centroid_y)]

    # Generate the binary image from roi
    binary_image = generate_binary_image(im_start, points)

     # Invert the binary image
    binary_image_inverted = cv2.bitwise_not(binary_image)

    # Create a white image of the same size as im_start
    white_image = np.ones_like(im_start) * 255

    # Resize the white image to match the size of im_start
    white_image_resized = cv2.resize(white_image, (im_start.shape[1], im_start.shape[0]))

    # Resize the binary image to match the size of im_start
    binary_image_inverted_resized = cv2.resize(binary_image_inverted, (im_start.shape[1], im_start.shape[0]))

    # Apply bitwise AND operation to make white pixels transparent
    im_start_masked = cv2.bitwise_and(im_start2, white_image_resized, mask=binary_image_inverted_resized)

    # Create a blended image by combining im_start and im_start_masked
    blended_image = cv2.add(im_start2, im_start_masked)

    # Display the blended image
    create_cv_window('Blended Image')
    cv2.imshow('Blended Image', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Display the binary image
    # create_cv_window('Binary Image')
    # cv2.imshow('Binary Image', binary_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Save the binary image
    cv2.imwrite(mask_file, binary_image)

    print(' ')
    print('Mask saved to: ' + mask_file)

    return centroid


def run_spatial_calibration(vid_file, reps=3, font_size=40):
    """Run spatial calibration.
    Args:
        vid_file (str): Video file.
        reps (int): Number of repetitions.
        font_size (int): Font size for the text.
    Returns:
        cm_per_pix (float): Centimeters per pixel.
    """

    # raise exception if video file does not exist
    if not os.path.isfile(vid_file):
        raise Exception('Video file does not exist: ' + vid_file)

    distances = []

    for i in range(reps):

        # Read frame of first video in cat_curr.video_filename
        vid = cv2.VideoCapture(vid_file)
        ret, im_start = vid.read()
        vid.release()

        # Perform spatial calibration
        title_text = 'Spatial Calibration: Image ' + str(i + 1) + ' of ' + str(reps)
        dist_pix = measure_length(im_start, title=title_text)
        distances.append(dist_pix)

    # Prompt user to enter the distance in cm
    dist_cm = prompt_number("Actual length", "Enter length in centimeters", font_size=font_size)

    # if len(dist_cm) == 0:
    #     raise Exception('No distance entered.')

    cm_per_pix = dist_cm / np.mean(distances)

    return cm_per_pix


def measure_length(frame, title='Measure Length'):
    """Select a polygon from an image.
    Args:
        frame (np.array): Image to select polygon from.
        title (str): Title to display above the image. Default is 'Measure Length'.
    Returns:
        points (list): List of points in the polygon.
    """

    # Global variables
    points = []

    # Make a copy of the frame
    frame_start = frame.copy()

    def display_image_points(frame, points):
        # Redraw the entire polygon
        frame[:] = frame_start.copy()
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        # Show the frame with the updated polygon
        cv2.imshow(title, frame)

    def draw_polygon(event, x, y, flags, param):
        nonlocal points

        if event == cv2.EVENT_LBUTTONDOWN:
            # Add the clicked point to the list
            if len(points) < 2:
                points.append((x, y))
            else:
                # Replace the last point with x, y
                points[-1] = (x, y)

            # Update image
            display_image_points(frame, points)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove the last point if right-clicked
            if len(points) > 0:
                # Remove the point from the list
                points.pop()

                # Update image
                display_image_points(frame, points)

    def calculate_distance(point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    # Create a window and set the mouse callback
    create_cv_window(title)
    cv2.setMouseCallback(title, draw_polygon)

    # Instructions
    print(f"Click to two points to measure length in '{title}'. Press 'Enter' to finish. Right-click to delete the last point.")

    # Display the initial frame with title
    cv2.imshow(title, frame)

    # Keep processing frames until 'Enter' is pressed
    while True:
        # Exit the loop if 'Enter' key is pressed
        if cv2.waitKey(1) == 13:  # 13 is the ASCII code for 'Enter' key
            break

    # Destroy the window
    cv2.destroyAllWindows()

    frame = frame_start.copy()

    # Calculate distance between points
    if len(points) == 2:
        distance = calculate_distance(points[0], points[1])
    else:
        raise ValueError('Must select two points to calculate distance.')

    # Return the polygon points
    return distance


def interactive_threshold(im, im_mean):
    """Apply interactive thresholding to identify darker regions.
    Args:
        im (np.array): Image to threshold.
        im_mean (np.array): Mean image.
    Returns:
        threshold (int): Threshold value.
        im_thresh (np.array): Thresholded image.
    """

    # Check that the images are grayscale
    dim_im = im.shape
    dim_immean = im_mean.shape
    if len(dim_im)>2:
        raise ValueError('Image must be grayscale.')
    elif len(dim_immean)>2:
        raise ValueError('Mean image must be grayscale.')

    # check that the dimensions of the two images are the same
    if dim_im != dim_immean:
        raise ValueError('Image and mean image must have the same dimensions.')
    
    # Name of figure window
    win_name = 'Select Threshold'

    print('Select a threshold value to identify darker regions. Press \'q\' or \'return\' to quit.')

    # Create a window to display the image
    create_cv_window(win_name)

    # Create a trackbar to adjust the threshold
    def on_trackbar_change(value):
        # Apply thresholding to the image
        im_thresholded = vp.threshold_diff_image(im, im_mean, value)

        # Display the thresholded image
        cv2.imshow(win_name, im_thresholded)

    # Set an initial threshold value
    initial_threshold = 20

    # Create a trackbar
    cv2.createTrackbar('Threshold',win_name, initial_threshold, 90, on_trackbar_change)

    # Initialize the thresholded image
    im_thresholded = vp.threshold_diff_image(im, im_mean, initial_threshold)

    # Display the initial thresholded image
    cv2.imshow(win_name, im_thresholded)

    # Wait until the user closes the window
    while True:
        key = cv2.waitKey(1) & 0xFF
        if (key == ord('q')) or (key==13):
            break

    # Retrieve the final threshold value
    threshold = int(cv2.getTrackbarPos('Threshold', win_name))

    cv2.destroyAllWindows()
    return threshold, im_thresholded


def interactive_blob_filter(im, im_mean, threshold):
    """Apply interactive blob filtering.
    Args:
        im (np.array): Image to threshold.
        im_mean (np.array): Mean image.
        threshold (int): Threshold value.
    Returns:
        min_area (int): Minimum area.
        max_area (int): Maximum area.
    """

    # Name of figure window
    win_name = 'Blob Filter'

    # Create a window to display the image
    create_cv_window(win_name)

    # Initialize the minimum and maximum blob sizes
    initial_min_area = 100
    initial_max_area = 1000

    # Create track bars for minimum and maximum blob sizes
    def on_min_area_change(value):
        # Get the current maximum blob size
        max_area = cv2.getTrackbarPos('Max Area', win_name)

        # Apply blob filtering to the image
        filtered_im = vp.filter_blobs(im, im_mean, threshold, value, max_area)

        # Display the filtered image
        cv2.imshow('Blob Filter', filtered_im)

    def on_max_area_change(value):
        # Get the current minimum blob size
        min_area = cv2.getTrackbarPos('Min Area', win_name)

        # Apply blob filtering to the image
        filtered_im = vp.filter_blobs(im, im_mean, threshold, min_area, value)

        # Display the filtered image
        cv2.imshow(win_name, filtered_im)

    # Create trackbars
    cv2.createTrackbar('Min Area', win_name, initial_min_area, 600, on_min_area_change)
    cv2.createTrackbar('Max Area', win_name, initial_max_area, 600, on_max_area_change)

    # Apply initial blob filtering to the image
    filtered_im = vp.filter_blobs(im, im_mean, threshold, initial_min_area, initial_max_area)

    # Display the initial filtered image
    cv2.imshow(win_name, filtered_im)

    # Wait until the user closes the window
    while True:
        key = cv2.waitKey(1) & 0xFF
        if (key == ord('q')) or (key==13):
            break

    # Retrieve the final minimum and maximum blob sizes
    min_area = cv2.getTrackbarPos('Min Area', win_name)
    max_area = cv2.getTrackbarPos('Max Area', win_name)

    cv2.destroyAllWindows()
    return min_area, max_area