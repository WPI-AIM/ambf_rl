#!/bin/bash

source /opt/ros/melodic/setup.bash
source $AMBF_WS/build/devel/setup.bash

rosrun dvrk_ambf_extensions psmIK_service.py