# |--- Maxim -----|
A Reachy Mini repo for creating home/office/workshop assistant capabilities.

# - Running Maxim

Must run Maxim from Reachy Mini currently by sshing into your Reachy in two terminals

`ssh pollen@<INSERT YOUR REACHY IP>`

Then enter the default password 'root' if first logging on or the unique password you
definitely reset it too.

clone this repo into a folder of your choosing

`git clone https://github.com/dennys246/Maxim.git`

Stop the process if something is using it.

`sudo systemctl stop reachy-mini-daemon`

Check to see if you can start a new process

`source /venvs/mini_daemon/bin/activate`

`python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only`


# Troubleshooting

1. Reachy Mini immendiately closing down on running or not running at all.

Check if the reachy mini port 8000 is occupied by ssh into you Reachy Mini then
checking if the port is occupied...

`ss -lntp | grep 8000`

Stop the process if something is using it.

`sudo systemctl stop reachy-mini-daemon`

Check to see if you can start a new process

`python -m reachy_mini.daemon.app.main --wireless-version --no-localhost-only`

