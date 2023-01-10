# wake_tracking

Code for running code through the wake-tracking project, which tests whether some fish species can follow the wake of neighboring fish in the dark. Here we have developed python code for controlling devices during the running of experiments and the acquisition of kinematic data from the video recordings.

# Virtual environment
You first need to install [Anaconda on your system](https://www.anaconda.com/products/distribution).
You will want to create a conda environment for executing the code. 
Start by setting up the environment, as instructured for [TRex](https://trex.run/docs/install.html).
In most situations, this amounts to running the following command at the terminal:

> conda create -n wake -c trexing trex

You will then want to active the new environment:

> conda activate wake

## Running experiments

For running experiments, you will need [DMXEnttecPro](https://github.com/SavinaRoja/DMXEnttecPro) for controlling a Enttec DMX USB Pro, which controls the lights. This may be installed as follows:

> pip install DMXEnttecPro timer plotly playsound multiprocess python-kasa asyncio

And the following:

> pip3 install PyObjC

<!-- And the following (for macs):

> brew install portaudio
> pip install pyaudio -->

## Acquiring kinematics

You will then want to install packages into that environment that are called by our code. For the kinematics, these packages can be installed as follows:

> pip install ipyparallel jupyter numpy matplotlib pandas

And the following:

>  conda install -c conda-forge opencv

For my M1 Mac, I had to install the [openblas package](https://stackoverflow.com/questions/70242015/python-how-to-solve-the-numpy-importerror-on-apple-silicon). 
After reinstalling numpy, I then had to reinstall TRex (conda install -c trexing trex).

# Hardware setup

## DMX controller for lights

We used a Luxli ORC-Taiko-2x1 LED light source controlled with an Enttex DMX USB Pro via a 5-pin DMX cable. Details on the Python package to control this hardware (DMXEnttecPro) can be found on [github](https://github.com/SavinaRoja/DMXEnttecPro).

You can first find the Enttec device by typing the folling at the terminal:

> python -m DMXEnttecPro.utils

which gave me the following:

>/dev/cu.usbserial-EN373474 <br>
>  name: cu.usbserial-EN373474 <br> 
>  description: DMX USB PRO<br>
>  hwid: USB VID:PID=0403:6001 SER=EN373474 LOCATION=0-1.1.1<br>
>  vid: 1027<br>
>  pid: 24577<br>
>  serial_number: EN373474<br>
>  location: 0-1.1.1<br>
>  manufacturer: ENTTEC<br>
>  product: DMX USB PRO<br>
>  interface: None<br>

The first line is what you'll want to copy and paste below to define 'hw_address', in [schooling_experiments.ipynb](schooling_experiments.ipynb).

## Smart switch for IR LED lights

As detailed in the instructions for the [python-kasa](https://python-kasa.readthedocs.io/en/latest/cli.html) package, you can get the IP address for the IR LED power strip at the command line, like this:

> kasa

Make sure that you are on the same wifi network as the smart switch and that your wifi connection is the only network connection when you run the above command (e.g., unplug ethernet). Once you have the IP address, then you can run the code with multiple network connections.

# Running experiments

Running experiments entails video-recording the swimming of fish under controlled lighting conditions.

- **[schooling_experiments.ipynb](schooling_experiments.ipynb)**: Jupyter notebook that steps through the running of experiments.

- **[def_runexperiments.py](def_runexperiments.py)**: Python functions called by schooling_experiments.ipynb to run experiments.


# Acquiring and analyzing kinematics

The execution of code is controlled by the "experiment_log" spreadsheet, which you will need to export from Google Sheets as a .csv file and save in the root directory for the project.

- **[acquire_kinematics.ipynb](acquire_kinematics.ipynb)**: Jupyter notebook that explains how to run the acquisition and includes the necessary code.


## Directory structure

The code assumes the following directory structure:

* "waketracking" [root_proj] - *Directory holding all videos and data.*
    * "experiment_log.csv" - *Experiment catalog, downloaded from google sheets.*
    * "data" 
        * "raw" - *Directory holding the data generated from the videos by TRex.*
        * "settings" - *Directory holding the settings files used by TRex.*
    * "video" 
        * "raw" - *Directory holding the recordings from experiments.*
            * (date) - *Directories for each date of experiments (e.g. '2022-10-03')*
        * "compressed" - *Code will generate compressed mp4 videos here.*
        * "tmp" - *Directory that generates temporary video files while compressing videos.*
        * "dv" - *dv-formatted videos, generated by TGrabs.*
    * "masks" - *Directory holding image files used for masking compressed videos.*    
* "wake_tracking" [root_code] - *Directory for the project code repository. Does not need to be anywhwere close to the "waketracking" [root_code].*
* "kineKit" - *Directory holding the kineKit repository. Must reside in same parent directory as "wake_tracking".*


## Acquisition 

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
