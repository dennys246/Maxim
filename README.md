# |------- Maxim ------- |

A Reachy Mini repo for orchestrating data streaming to and from a PC and Reachy Mini to orchestrate agents and models.

# - Running Maxim

Run the Reachy Mini daemon on the robot, then run `Maxim` from any computer on the same LAN/Wi‑Fi (Zenoh peer discovery).

`ssh pollen@<INSERT YOUR REACHY IP>`

Then enter the default password 'root' if first logging on or the unique password you reset it too.

Stop the process if something is using it.

`sudo systemctl stop reachy-mini-daemon`

Check to see if you can start a new daemon process

`source /venvs/mini_daemon/bin/activate`

`python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only`

On your controller computer clone this repo into a folder of your choosing

`git clone https://github.com/dennys246/Maxim.git`

Prepare a computing environment for running Maxim by creating a new python virtual environment.
Avoid installing requirements into a virtual environment you typically use for machine learning
as it may mess up your tensorflow or pytorch dependencies and how your GPU is handled.

```
python -m venv maxim-env

cd Maxim
pip install requirements.py
```

Run the main command in the Maxim directory to initiate basic observation using Ultralytics
incredibly efficient YOLO8 model. This dynamically find objects of interest and center the 
Reach Mini vision on them. Audio currently isn't handle yet.

`python main.py --robot-name reachy_mini`

NOTE: You can also set `MAXIM_ROBOT_NAME=reachy_mini` and run `python main.py`.

You can also run Maxim straight from a python shell of your own script by importing it

```
from src.conscience.selfy import Maxim

maxim = Maxim()

# This calls to the YOLO8 guided observation of things of interest
maxim.observe()

# General movement wrapper to the Reachy SDK
maxim.move(y = 10, yaw = 3)

```

Of course extensions of the Maxim class using the datastreams set up are more than welcomed!

```
from src.conscience.selfy import Maxim

Maxim

```

For easy future use consider editing your Reachy's .bashrc...

`nano ~/.bashrc`

and adding aliases so you can run simple commands to start processes

```
alias mini-env='source /venvs/mini_daemon/bin/activate'

alias list-daemon='ss -lntp | grep 8000'
alias clear-daemon='sudo systemctl stop reachy-mini-daemon'
alias start-daemon='python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only'

alias list-zenoh `ss -lntp | grep 7447`

MAXIM_ROBOT_NAME=reachy_mini
REACHY_IP=<INSERT YOUR REACHY IP>
```
Then you can simply type commands like list-daemon, clear-daemon or start-daemon.

# Networking

Make sure you are on the same network as your Reachy Mini with no VPN. With a VPN you 
may be able to do simple things like start the daemon but the python SDK will
struggle to connect to the Reachy.


# Troubleshooting

1. Reachy Mini immendiately closing down on running or not running at all.

Check if the reachy mini port 8000 is occupied by ssh into you Reachy Mini then
checking if the port is occupied...

`ss -lntp | grep 8000`

Stop the process if something is using it.

`sudo systemctl stop reachy-mini-daemon`

Check to see if you can start a new process

`python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only`
