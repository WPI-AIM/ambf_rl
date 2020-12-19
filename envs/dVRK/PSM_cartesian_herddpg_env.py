#!/usr/bin/env python

# //============================================================================
# /*
#     Software License Agreement (BSD License)
#     Copyright (c) 2019, AMBF
#     (www.aimlab.wpi.edu)

#     All rights reserved.

#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions
#     are met:

#     * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.

#     * Neither the name of authors nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.

#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#     FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#     COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#     INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#     BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#     ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#     POSSIBILITY OF SUCH DAMAGE.

#     \author    <http://aimlab.wpi.edu>
#     \author    <dkoolrajamani@wpi.edu>, <vvarier@wpi.edu>, <amunawar@wpi.edu>
#     \author    Dhruv Kool Rajamani, Vignesh Manoj Varier, and Adnan Munawar
#     \version   0.1.0
# */
# //============================================================================

import sys, os, copy, time

from typing import Iterable, List, Set, Tuple, Dict, Any, Type
from arl.arl_env import Action, Goal, Observation, ARLEnv
from .PSM_cartesian_env import PSMCartesianEnv, CartesianAction

import numpy as np
from numpy import linalg as LA

import gym
from gym import spaces
from gym.utils import seeding

from psmFK import compute_FK
from transformations import euler_from_matrix
from dvrk_functions.msg import HomogenousTransform
import rospy
from dvrk_functions.srv import *


class HERDDPGObservation(Observation):

  def __init__(
    self,
    state: Dict,
    dist: int = 0,
    reward: float = 0.0,
    prev_reward: float = 0.0,
    cur_reward: float = 0.0,
    is_done: bool = False,
    info: Dict = {},
    sim_step_no: int = 0
  ) -> None:

    super(HERDDPGObservation,
          self).__init__(state,
                         dist,
                         reward,
                         prev_reward,
                         cur_reward,
                         is_done,
                         info,
                         sim_step_no)
    return


class PSMCartesianHERDDPGEnv(PSMCartesianEnv, gym.GoalEnv):
  """Single task based environment for PSM to perform debris removal as shown in
  the paper:
  
  TODO: Enter paper doi and name

  The environment performs actions in the cartesian space in R3, with translational
  movements in x, y, and z.
  """

  def __init__(
    self,
    action_space_limit: float,
    goal_position_range: float,
    position_error_threshold: float,
    goal_error_margin: float,
    joint_limits: Dict[str,
                       np.ndarray or List[str]],
    workspace_limits: Dict[str,
                           np.ndarray or List[str]],
    enable_step_throttling: bool,
    joints_to_control: List[str] = [
      'baselink-yawlink',
      'yawlink-pitchbacklink',
      'pitchendlink-maininsertionlink',
      'maininsertionlink-toolrolllink',
      'toolrolllink-toolpitchlink',
      'toolpitchlink-toolgripper1link',
      'toolpitchlink-toolgripper2link'
    ],
    steps_to_print: int = 10000,
    n_actions: int = 3,
    n_skip_steps: int = 5,
    env_name: str = "PSM_cartesian_ddpg_env"
  ) -> None:
    """Initialize an object to train with DDPG on the PSM robot.

    Parameters
    ----------
    action_space_limit : float
        Action space limit for cartesian actions
    goal_position_range : int, optional
        The variance in goal position
    position_error_threshold : float
        Maximum acceptable error in cartesian position
    goal_error_margin : float
        Maximum margin of error for an epoch to be considered successful
    joint_limits : Dict(str, List(float) | np.array(float))
        Robot joint limits in radians
    workspace_limits : Dict(str, List(float) | np.array(float))
        Workspace limits in x,y, and z for the robots workspace in Cartesian space
    enable_step_throttling : bool
        Flag to enable throttling of the simulator
    joints_to_control : np.array(str) | List(str)
        The list of joint links for the psm.
    steps_to_print : int
        Number of steps before model prints information to stdout
    n_actions : int
        Number of possible actions
    n_skip_steps : int
        Number of steps to skip after an update step
    env_name : str
        Name of the environment to train
    """
    super(PSMCartesianHERDDPGEnv,
          self).__init__(
            action_space_limit,
            goal_position_range,
            position_error_threshold,
            goal_error_margin,
            joint_limits,
            workspace_limits,
            enable_step_throttling,
            joints_to_control,
            steps_to_print,
            n_actions,
            n_skip_steps,
            env_name
          )

    # Set observation space and constraints
    self.obs = Observation(
      state={
        'observation': np.zeros(20),
        # Prismatic joint is set to 0.1 to ensure at least some part of robot tip goes past the cannula
        'achieved_goal': np.array([0.0,
                                   0.0,
                                   0.1,
                                   0.0,
                                   0.0,
                                   0.0]),
        'desired_goal': np.zeros(6)
      }
    )
    self.initial_pos = copy.deepcopy(self.obs.cur_observation()[0])
    self.observation_space = spaces.Dict(
      dict(
        desired_goal=spaces.Box(
          -np.inf,
          np.inf,
          shape=self.initial_pos['achieved_goal'].shape,
          dtype='float32'
        ),
        achieved_goal=spaces.Box(
          -np.inf,
          np.inf,
          shape=self.initial_pos['achieved_goal'].shape,
          dtype='float32'
        ),
        observation=spaces.Box(
          -np.inf,
          np.inf,
          shape=self.initial_pos['observation'].shape,
          dtype='float32'
        ),
      )
    )

    return


if __name__ == "__main__":
  # Create object of this class
  root_link = 'baselink'
  env_kwargs = {
    'action_space_limit': 0.05,
    'goal_position_range': 0.05,
    'position_error_threshold': 0.01,
    'goal_error_margin': 0.0075,
    'joint_limits':
      {
        'lower_limit': np.array([-0.2,
                                 -0.2,
                                 0.1,
                                 -1.5,
                                 -1.5,
                                 -1.5,
                                 -1.5]),
        'upper_limit': np.array([0.2,
                                 0.2,
                                 0.24,
                                 1.5,
                                 1.5,
                                 1.5,
                                 1.5])
      },
    'workspace_limits':
      {
        'lower_limit': np.array([-0.04,
                                 -0.03,
                                 -0.2]),
        'upper_limit': np.array([0.03,
                                 0.04,
                                 -0.091])
      },
    'enable_step_throttling': False,
  }
  psmEnv = PSMCartesianHERDDPGEnv(**env_kwargs)
  psmEnv.make(root_link)
  # psmEnv.world_handle = psmEnv.ambf_client.get_world_handle()
  # psmEnv.world_handle.enable_throttling(False)