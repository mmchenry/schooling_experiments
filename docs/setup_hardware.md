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