#%% =================================================================================================
""" Parameters and packages """
# - You'll need to execute the next cell for all of the code that follows.
# - Make sure that the root_path and local_path are correct for your system.
#     - The root_path is the path to the folder containing the data, and video folders.
# - local_path needs to be a directory on a local drive for writing binary video files for TGrabs/TRex.

# The project name need to match a directory name within the root path
proj_name = 'RN_Scale'
# proj_name = 'RN_Prop'

# This specifies whether the mask is specific to a trial (True) or the same for all trials (False)
trial_specific_mask = False

# Other details about the project
species = 'rummy_nose'
exp_type = 'scaling'

# font size for GUIs
font_size = 30

# Repeated measures for each calibration video
num_reps = 3

# Max number of frames for the mean image
max_num_frame_meanimage = 200

# Raw and processed video extensions
vid_ext_raw = 'MOV'
vid_ext_proc = 'mp4'

# Installed packages
import os
import platform

# Our own modules
import def_acquisition as da
import def_paths as dp
import video_preprocess as vp
import acqfunctions as af
import gui_functions as gf

# DEFINE ROOT PATH ============================================================

# Matt's laptop
if (platform.system() == 'Darwin') and (os.path.expanduser('~')=='/Users/mmchenry'):
    
    root_path = '/Users/mmchenry/Documents/Projects/waketracking'

# Matt on PopOS! machine
elif (platform.system() == 'Linux') and (os.path.expanduser('~')=='/home/mmchenry'):

    # root_path = '/home/mmchenry/Documents/wake_tracking'
    root_path = '/mnt/schooling/TRex'
    local_path = '/home/mmchenry/Documents/wake_tracking/video/binary'

# Ashley on Linux
elif (platform.system() == 'Linux') and (os.path.expanduser('~')=='/home/anpetey'):

    root_path = '/vortex/schooling/TRex'
    local_path = '/home/anpetey/Documents/wake_tracking/video/binary'

# Catch alternatives
else:
    raise ValueError('Do not recognize this account -- add lines of code to define paths here')

# =============================================================================

# Check for local path definition
if not 'local_path' in locals():
    raise ValueError('Local path not defined')

# Check paths
if not os.path.exists(root_path):
    raise ValueError('Root path does not exist: ' + root_path)
elif not os.path.exists(local_path):
    raise ValueError('Local path does not exist: ' + local_path)

# Get paths 
path = dp.give_paths(root_path, proj_name)


#%% =================================================================================================
""" Select schedule, check for problems in recordings"""
# Note: need to run this for cells below.
#     Here we prompt the user to select which schedule to choose for preprocessing. Along the way, it checks for the following:
#     - That the experiment_log.csv and recording_log.csv lists include all trials in the schedule.
# - Compares the schedules in the project against video recordings
# - It compares the duration of recorded videos to what was expected in the schedule and alerts user of large differences.

# Find matching experiments between the schedule and video directories
sch_num, sch_date, analysis_schedule = vp.find_schedule_matches(path['sch'], path['vidin'], font_size=30)

# Check that the schedule matches the catalog and the catalog matches the experiment log. Also check that the video files exist. Add timecode data.
vp.check_logs(path, analysis_schedule, sch_num, sch_date, vid_ext_raw)

#%% 
# This cell left empty on purpose



#%% =================================================================================================
""" Create a mask image"""
# You will want to choose a region of interest that is just outside of the water line within the arena.

gf.run_mask_acq(path, sch_date, sch_num, vid_ext_raw, analysis_schedule, overwrite_existing=True, 
                trial_specific_mask=trial_specific_mask)


#%% =================================================================================================
""" Run spatial calibration """
# Prompts user to conduct repeated measures for the calibration. Note that you need to know the actual length in centimeters.

gf.run_spatial_calibration(path, sch_date, sch_num, vid_ext_raw, analysis_schedule, num_reps, font_size=40,
                           overwrite_existing=True)


#%% =================================================================================================
""" Create mean image"""
# A mean image is created from multiple videos in the batch.

vp.run_mean_image(path, sch_num, sch_date, analysis_schedule, max_num_frame_meanimage, overwrite_existing=True,
                  trial_specific_mean=trial_specific_mask, show_image=False)


#%% =================================================================================================
""" Select threshold and blob area"""
# - Select the lowest threshold possible, without the margins of each fish looking fuzzy
# - Select the range of areas that just barely include individual fish. Exclude fish that are touching area other.

gf.run_threshold_choice(path, sch_date, sch_num, analysis_schedule, vid_ext_raw,  overwrite_existing=True)


#%% =================================================================================================
""" Generate binary videos """
# - Here we use the threshold and area values to generate black-and-white images of the school.
# - This can be performed on a single video, all videos in a schedule in succession, or using parallel processing
# (the fastest option).

run_mode = 'sequential' # May be single, sequential, or parallel
vp.run_make_binary_videos(run_mode, path, local_path, proj_name, vid_ext_raw, vid_ext_proc, mov_idx=0)


#%% =================================================================================================
""" TGrabs/TRex Parameters """
# - Parameters are described in the documentation for [TGrabs](https://trex.run/docs/parameters_tgrabs.html) and
# [TRex](https://trex.run/docs/parameters_trex.html).
# - These lists of parameters will be passed to TGrabs and TRex. If there is not already a column for that parameter,
#     then it will be added to cat (i.e. experiment_log.csv) with the default values specified below.
# Those defaults may be overridden by keying values into experiment_log.csv.

# Parameter list to use by TGrabs, along with default
param_list_tgrabs = [
    #['threshold','20'],
    ['averaging_method','mode'],
    ['average_samples','150'],
    ['blob_size_range','[0.01,5]'],
    ['meta_conditions',exp_type],
    ['meta_species',species]
   # ['meta_misc','school_ABC']
    ]
# Specify list of parameter values for TRex, all listed as strings
param_list_trex = [
    ['track_threshold','20'],
    ['blob_size_ranges','[0.01,3]'],
    ['track_max_speed','70'],
    ['track_max_reassign_time','0.1'],
    ['output_format','npz'],
    ['gui_show_posture','false'],
    ['gui_show_paths','false'],
    ['gui_show_outline', 'true'], 
    ['gui_show_midline', 'false'], 
    ['gui_show_blobs', 'true'],
    ['calculate_posture','true'],
    ['gui_show_number_individuals', 'true']
    ]

# Map 'cat' column names to TRex parameter names (no default values)
cat_to_trex = [
    ['fish_num','track_max_individuals'],
    ['cm_per_pix','cm_per_pixel'],
    ['frame_rate','frame_rate']
    ]

# Add default parameter values to all rows
af.add_param_vals(path['cat'], param_list_tgrabs, param_list_trex)


#%% =================================================================================================
""" Run TGrabs"""
# - TGrabs generates pv video files from raw videos for TRex tracking.
#     - Cell below generates dv videos, which will be used by TRex, from compressed videos.
# - This will be completed for each row of cat.

# Run TGrabs, or formulate the command-line terminal commands
commands = af.run_tgrabs(path['cat'], path['data_raw'], local_path + os.sep + proj_name, path['vidpv'],
                         param_list_tgrabs, vid_ext_proc=vid_ext_proc, use_settings_file=False, run_gui=True,
                         echo=True, run_command=True)


#%% =================================================================================================
""" Run TRex """
# - Uses the parameter names given in param_list_trex and cat_to_trex to generate the command-line terminal commands to run TRex.

# Run TRex, or formulate the command-line terminal commands
commands = af.run_trex(path['cat'], path['vidpv'], path['data_raw'], param_list_trex, cat_to_trex,
                run_gui=True, auto_quit=True, output_posture=True, echo=True, run_command=True)


#%% =================================================================================================
""" Export TRex data in mat format"""

# Extract experiment catalog info
cat = af.get_cat_info(path['cat'], include_mode='matlab', exclude_mode='calibration')

# Convert all npz files for an experiment to mat files.
da.raw_to_mat(cat, path)


#%% =================================================================================================
""" Housecleaning """

# Delete the local binary videos that match the pv videos
af.delete_matching_files(local_path, path['vidpv'])