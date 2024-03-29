# schooling_experiments

Code for running code through the wake-tracking project, which tests whether some fish species can follow the wake of neighboring fish in the dark. Here we have developed python code for controlling devices during the running of experiments and the acquisition of kinematic data from the video recordings.

# Setup
Read the following pages for tips on setting up the software and hardware for running experiments.

- **[Software setup](docs/setup_software.md)** 
- **[Hardware setup](docs/setup_hardware.md)** 


# Running experiments

Running experiments entails video-recording the swimming of fish under controlled lighting conditions.

- **[schooling_experiments.ipynb](schooling_experiments.ipynb)**: Jupyter notebook that steps through the running of experiments.

- **[def_runexperiments.py](def_runexperiments.py)**: Python functions called by schooling_experiments.ipynb to run experiments.


## Acquiring video

The execution of code is controlled by the "experiment_log" spreadsheet, which you will need to export from Google Sheets as a .csv file and save in the root directory for the project.

- **[acquire_kinematics.ipynb](acquire_kinematics.ipynb)**: Jupyter notebook that explains how to run the acquisition and includes the necessary code.


### Directory structure

The code assumes the following directory structure. This will be self-generated when running code in [schooling_experiments.ipynb](schooling_experiments.ipynb).

* "waketracking" [root_proj] - *Directory holding all videos and data.*
    * "data"
        * [project directory] - *Named after the project name, specified in schooling_experiments.ipynb.*
            * "experiment_log.csv" - *Experiment catalog, downloaded from google sheets.*
            * "data" 
                * "raw" - *Directory holding the data generated from the videos by TRex.*
                  * "fishdata" - *Directory holding the data generated from the videos by TRex.*
                * "settings" - *Directory holding the settings files used by TRex.*
                * "matlab" - *mat files of the TRex data, analyzed in Matlab.*
            * experiment_schedules - *csv files generated to control the camera and lights on a schedule.*
            * masks - *Image files generated by the code to generate a mask over the videos.*
            * calibration_images - *Image files generated for measuring the calibration constant.*
    * "video" 
        * [project directory] - *Named after the project name, specified in schooling_experiments.ipynb.*
            * "raw" - *Directory holding the recordings from experiments.*
                * (date) - *Directories for each date of experiments (e.g. '2022-10-03')*
            * "compressed" - *Code will generate compressed mp4 videos here.*
            * "calibration" - *Generated from the raw videos for measuring calibration constant.*
            * "tmp" - *Directory that generates temporary video files while compressing videos.*
            * "dv" - *dv-formatted videos, generated by TGrabs.*


### Acquisition 

Includes the following files: 

- **[acquire_kinematics.ipynb](acquire_kinematics.ipynb)**: Jupyter notebook that explains how to run the acquisition and includes the necessary code.

- **[def_paths.py](def_definepaths.py)**: Defines the data and video paths for the project. You need to add root paths for each new user or machine included in the project (and push the addition).

- **[def_acquisition.py](def_acquisition.py)**: Functions for running data acquisition.

- **[acqfunctions.py](acqfunctions.py)**: Copied from [kineKit](https://github.com/mmchenry/kineKit), these functions are used to prep video for TRex.

- **[videotools.py](videotools.py)**: Copied from [kineKit](https://github.com/mmchenry/kineKit), series of functions for manipulating and interacting with video. Requires installing ffmpeg and opencv.

[//]: # ()
[//]: # (## Processing)

[//]: # (Taking the raw coordinates from DLC videos, cleaning the data, and generating parameter metrics of the kinematics. Controlled with runProcessing.)

[//]: # ()
[//]: # (## Analysis)

[//]: # (Exploratory data analysis. )

[//]: # ()
[//]: # (## Presentation )

[//]: # (Stats and final figure construction.)
