#!/bin/bash

source /opt/ros/melodic/setup.bash
source $AMBF_WS/build/devel/setup.bash

pushd $AMBF_WS/bin/lin-x86_64/
./ambf_simulator -l 5,22 -p 2000 -t 1 -s 2 # -g false
popd
