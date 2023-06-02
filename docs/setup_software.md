
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