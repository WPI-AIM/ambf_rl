#!/bin/bash
source /opt/ros/melodic/setup.bash
source $AMBF_WS/build/devel/setup.bash

python $AMBF_RL_WS/scripts/dVRK/PSM_cartesian_herddpg_algorithm.py
