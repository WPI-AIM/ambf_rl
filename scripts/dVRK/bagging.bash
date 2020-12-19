#!/bin/bash

arg=$1

rosbag record -e "/ambf/env/(.*)/(.*)/Command" -O $arg
