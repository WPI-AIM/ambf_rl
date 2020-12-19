#!/bin/bash
source /opt/ros/melodic/setup.bash
source $AMBF_WS/build/devel/setup.bash

python $ARL_WS/scripts/dVRK/PSM_cartesian_ddpg_algorithm.py
