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
#     \author    <amunawar@wpi.edu>, <vvarier@wpi.edu>, <dkoolrajamani@wpi.edu>
#     \author    Adnan Munawar, Vignesh Manoj Varier, and Dhruv Kool Rajamani
#     \version   0.1
# */
# //============================================================================

from typing import List, Set, Tuple, Dict, Any
from ambf_client import Client
from ..arl.arl_env import Action, Goal, Observation, ARLEnv
import numpy as np
import copy
import time
import gym
import sys, os
from gym.utils import seeding


class PSMCartesianDDPGEnv(ARLEnv):
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
                       Any or List[str]],
    workspace_limits: Dict[str,
                           Any or List[str]],
    enable_step_throttling: bool,
    joints_to_control: Any or List[str] = [
      'baselink-yawlink',
      'yawlink-pitchbacklink',
      'pitchendlink-maininsertionlink',
      'maininsertionlink-toolrolllink',
      'toolrolllink-toolpitchlink',
      'toolpitchlink-toolgripper1link',
      'toolpitchlink-toolgripper2link'
    ],
    n_actions: int = 3,
    n_skip_steps: int = 5,
    env_name="PSM_cartesian_ddpg_env"
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
    n_actions : int
        Number of possible actions
    n_skip_steps : Number of steps to skip after an update step
    env_name : str
        Name of the environment to train
    """
    super(PSMCartesianDDPGEnv, self).__init__(enable_step_throttling, n_skip_steps, env_name)

    # Set environment limits
    self._position_error_threshold = position_error_threshold
    self._joint_limits = joint_limits
    self._workspace_limits = workspace_limits

    # Store controllable joints
    self._joints_to_control = joints_to_control

    # Set environment and task parameters
    self._n_actions = n_actions

    ## Set task constraints
    # Set action space limits
    self.action = Action(self._n_actions, action_space_limit)

    # Set observation space and constraints
    self.obs = Observation(state=np.zeros(20))
    self._initial_pos = copy.deepcopy(self.obs.cur_observation()[0])

    # Set goal position and constraints
    # TODO: Consider using args/extra args in init() to specify goal.
    self.goal = Goal(
      goal_position_range,
      goal_error_margin,
      np.array([0.0,
                0.0,
                -0.1,
                0.0,
                0.0,
                0.0])
    )

    return


# Properties

  @property
  def position_error_threshold(self) -> float:
    """Returns the position error threshold
    """
    return self._position_error_threshold

  @position_error_threshold.setter
  def position_error_threshold(self, value: float):
    """Set the position error threshold
    """
    self._position_error_threshold = value
    return

  @property
  def joint_limits(self) -> Dict[str, Any or List[float]]:
    """Return the joint limits dictionary
    """
    return self._joint_limits

  @joint_limits.setter
  def joint_limits(self, value: Dict[str, Any or List[float]]):
    """Set the joint limits dictionary
    """
    self._joint_limits = value
    return

  

  # Overriding base gym functions
  def reset(self):
    pass
