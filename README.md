# wake_tracking

Code for running code through the wake-tracking project, which tests whether some fish species can follow the wake of neighboring fish in the dark. The execution of code is controlled by the "experiment_log" spreadsheet, which you will need to export from Google Sheets as a .csv file and save in the 

# Directory structure

The code assumes the following directory structure:

* "waketracking" [root_proj] - *Directory holding all videos and data.*
    * "experiment_log.csv" - *Experiment catalog, downloaded from google sheets.*
    * "data" - *Directory holding the data generated from the videos.*
    * "video" - *Directory holding video files.*
        * "raw" - *Directory holding the recordings from pilot experiments.*
        * "compressed" - *Code will generate compressed mp4 videos here.*
        * "tmp" - *Directory that generates temporary video files while compressing videos.*
    * "masks" - *Directory holding image files used for masking compressed videos.*    
* "wake_tracking" [root_code] - *Directory for the project code repository. Does not need to be anywhwere close to the "waketracking" [root_code].*
* "kineKit" - *Directory holding the kineKit repository. Must reside in same parent directory as "wake_tracking".*

Note that when this is set up on a new machine (or new user), you need to add and push code to the 'give_paths' function in the def_definepaths.py file, within the wake_tracking directory. This will allow the code to find all of the data and video files for the project.


# Acquisition

Includes the following files: 

- **[acquire_kinematics.ipynb](acquire_kinematics.ipynb)**: Jupyter notebook that explains how to run the acquisition and includes the necessary code.

- **[run_acquisition.py](run_acquisition.py)**: This is a coders version for developing the acquisition code. Includes some dead-ends not included in acquire_kinematics.ipynb .

- **[def_definepaths.py](def_definepaths.py)**: Defines the data and video paths for the project. You need to add root paths for each new user or machine included in the project (and push the addition).

- **[def_acquisition.py](def_acquisition.py)**: Functions for running data acquisition.



[//]: # ()
[//]: # (## Processing)

[//]: # (Taking the raw coordinates from DLC videos, cleaning the data, and generating parameter metrics of the kinematics. Controlled with runProcessing.)

[//]: # ()
[//]: # (## Analysis)

[//]: # (Exploratory data analysis. )

[//]: # ()
[//]: # (## Presentation )

[//]: # (Stats and final figure construction.)
