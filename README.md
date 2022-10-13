# wake_tracking

Code for running code through the wake-tracking project, which tests whether some fish species can follow the wake of neighboring fish in the dark. 
The execution of code is controlled by the "experiment_log" spreadsheet, which you will need to export from Google Sheets as a .csv file and save in the root directory for the project.

# Virtual environment

You will want to create a conda environment for executing the code. 
Start by setting up the environment, as instructured for [TRex](https://trex.run/docs/install.html).
In most situations, this amounts to running the following command at the terminal:

> conda create -n tracking -c trexing trex

You will then want to active the new environment:

> conda activate tracking

You will then want to install packages into that environment that are called by our code. These can be installed as follows:

> pip install ipyparallel jupyter numpy matplotlib pandas

And the following:

>  conda install -c conda-forge opencv

For my M1 Mac, I had to install the [openblas package](https://stackoverflow.com/questions/70242015/python-how-to-solve-the-numpy-importerror-on-apple-silicon). 
After reinstalling numpy, I then had to reinstall TRex (conda install -c trexing trex).

# Directory structure

The code assumes the following directory structure:

* "waketracking" [root_proj] - *Directory holding all videos and data.*
    * "experiment_log.csv" - *Experiment catalog, downloaded from google sheets.*
    * "data" - *Directory holding the data generated from the videos.*
    * "video" - *Directory holding video files.*
        * "raw" - *Directory holding the recordings from pilot experiments.*
        * "compressed" - *Code will generate compressed mp4 videos here.*
        * "tmp" - *Directory that generates temporary video files while compressing videos.*
        * "dv" - *dv-formatted videos, generated by TGrabs.*
    * "masks" - *Directory holding image files used for masking compressed videos.*    
* "wake_tracking" [root_code] - *Directory for the project code repository. Does not need to be anywhwere close to the "waketracking" [root_code].*
* "kineKit" - *Directory holding the kineKit repository. Must reside in same parent directory as "wake_tracking".*


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
