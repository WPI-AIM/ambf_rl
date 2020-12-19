#!/usr/bin/env python

from distutils.core import setup

setup(
  name='AMBF_RL',
  version='0.1.0',
  description='AMBF Reinforcement Learning Toolkit',
  author='Dhruv Kool Rajamani',
  author_email='dkoolrajamani@wpi.edu',
  package_dir={'': 'envs'},
  packages=['arl',
            'dVRK']
)
