""" Copies over all data files and experiment_log.csv files from the TRex on Vortex to a local directory. """

import os
import shutil

# Source and destination root directories
src_root = "/Volumes/schooling/TRex/data"
dest_root = "/Users/mmchenry/Documents/Projects/waketracking/data"

# Directories to exclude
excluded_dirs = ['.DS_Store', 'Test', "TestSchoolBhav", "miniScaleRN", "RN_Prop_prelim", "Test", "TestSchoolBehav", "rampTest", "RN_Ramp_Debug","blank_settings.settings", "_RN_Scale"]

# Files to include, within the matlab/centroid folder
included_files = ['_rawfish.mat', '_mutual_info.mat', '_peaks.mat', '_network.mat', '_schooldata.mat', '_focalfish.mat', '_peaks.mat']

# Get list of project directories, excluding unwanted ones
project_dirs = [dir for dir in os.listdir(src_root) if dir not in excluded_dirs and os.path.isdir(os.path.join(src_root, dir))]
total_projects = len(project_dirs)

# Iterate through project directories in the source root
for idx, project_dir in enumerate(project_dirs):
    print(f"Processing {idx+1}/{total_projects}: {project_dir}")
    
    src_project_path = os.path.join(src_root, project_dir)
    dest_project_path = os.path.join(dest_root, project_dir)
    
    # Make sure the destination project directory exists
    if not os.path.exists(dest_project_path):
        os.makedirs(dest_project_path)
    
    # Copy 'experiment_log.csv'
    src_csv = os.path.join(src_project_path, 'experiment_log.csv')
    dest_csv = os.path.join(dest_project_path, 'experiment_log.csv')
    if os.path.isfile(src_csv):
        shutil.copy(src_csv, dest_csv)
    
    # Copy files in 'matlab/centroid' ending with specific patterns
    src_matlab_path = os.path.join(src_project_path, 'matlab', 'centroid')
    dest_matlab_path = os.path.join(dest_project_path, 'matlab', 'centroid')
    
    # MATLAB FILES
    if os.path.exists(src_matlab_path):
        # Create destination matlab/centroid folder if it doesn't exist
        if not os.path.exists(dest_matlab_path):
            os.makedirs(dest_matlab_path)
        
        # Iterate through all files and copy the ones that match our criteria
        for filename in os.listdir(src_matlab_path):
            # if filename.endswith(('_rawfish.mat','mutual_info.mat','_peaks.mat','_networ.mat','_schooldata.mat', '_focalfish.mat')):
            if filename.endswith(included_files):
                src_file = os.path.join(src_matlab_path, filename)
                if os.path.isfile(src_file):
                    dest_file = os.path.join(dest_matlab_path, filename)
                    shutil.copy(src_file, dest_file)

    # Copy files from 'masks' 
    src_masks_path = os.path.join(src_project_path, 'masks')
    dest_masks_path = os.path.join(dest_project_path, 'masks')

    # MASK FILES
    if os.path.exists(src_masks_path):
        # Create destination masks folder if it doesn't exist
        if not os.path.exists(dest_masks_path):
            os.makedirs(dest_masks_path)

        # Iterate through all files and copy the ones that match our criteria
        for filename in os.listdir(src_masks_path):
            src_file = os.path.join(src_masks_path, filename)
            if os.path.isfile(src_file):
                dest_file = os.path.join(dest_masks_path, filename)
                shutil.copy(src_file, dest_file)
