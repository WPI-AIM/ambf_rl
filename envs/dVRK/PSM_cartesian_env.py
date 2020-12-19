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
#     \version   1.0.0
# */
# //============================================================================

from abc import ABCMeta, abstractmethod
import sys, os, copy, time

from typing import Iterable, List, Set, Tuple, Dict, Any, Type

from arl.arl_env import Action, Goal, Observation, ARLEnv

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


class CartesianAction(Action):

  def __init__(
    self,
    n_actions: int,
    action_space_limit: float,
    action_lims_low: List[float] = None,
    action_lims_high: List[float] = None
  ) -> None:
    super(CartesianAction,
          self).__init__(n_actions,
                         action_space_limit,
                         action_lims_low,
                         action_lims_high)

    self.action_space = spaces.Box(
      low=-action_space_limit,
      high=action_space_limit,
      shape=(self.n_actions,
             ),
      dtype="float32"
    )


class PSMCartesianEnv(ARLEnv, metaclass=ABCMeta):
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
    env_name: str = "PSM_cartesian_ddpgenv"
  ) -> None:
    """Initialize an object of the PSM robot in cartesian space.

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
    super(PSMCartesianEnv, self).__init__(enable_step_throttling, n_skip_steps, env_name)

    # Set environment limits
    self._position_error_threshold = position_error_threshold
    self._joint_limits = joint_limits
    self._workspace_limits = workspace_limits

    # Store controllable joints
    self._joints_to_control = joints_to_control

    # Steps to print
    self._steps_to_print = steps_to_print

    # Set environment and task parameters
    self._n_actions = n_actions

    ## Set task constraints
    # Set action space limits
    self.action = CartesianAction(self._n_actions, action_space_limit)
    self.action_space = self.action.action_space

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
  def joint_limits(self) -> Dict[str, np.ndarray or List[float]]:
    """Return the joint limits dictionary
    """
    return self._joint_limits

  @joint_limits.setter
  def joint_limits(self, value: Dict[str, np.ndarray or List[float]]):
    """Set the joint limits dictionary
    """
    self._joint_limits = value
    return

  @property
  def workspace_limits(self) -> Dict[str, np.ndarray or List[float]]:
    """Returns the workspace limits dictionary
      """
    return self._workspace_limits

  @workspace_limits.setter
  def workspace_limits(self, value: Dict[str, np.ndarray or List[float]]):
    """Set the workspace limits dictionary
    """
    self._workspace_limits = value
    return

  @property
  def joints_to_control(self) -> Any or List[str]:
    """Returns a np.array or List of joint object handles
    """
    return self._joints_to_control

  @joints_to_control.setter
  def joints_to_control(self, value: Any or List[str]):
    """Set a np.array or List of joint object handles
    """
    self._joints_to_control = value
    return

  @property
  def steps_to_print(self) -> int:
    """Returns the number of steps to print
    """
    return self._steps_to_print

  @steps_to_print.setter
  def steps_to_print(self, value: int):
    """Set the number of steps to print
    """
    self._steps_to_print = value
    return

  @property
  def n_actions(self) -> int:
    """Returns the number of actions possible
    """
    return self._n_actions

  @n_actions.setter
  def n_actions(self, value: int):
    """Set the number of actions possible
    """
    self._n_actions = value
    return

  @property
  def initial_pos(self) -> Any or np.ndarray or List[float] or Dict:
    """Returns the initial position (state) of the environment
    """
    # if type(self._initial_pos) == type(np.ndarray):
    #   return self._initial_pos
    # elif type(self._initial_pos) == type(dict):
    #   return self._initial_pos
    # else:
    #   return np.array(self._initial_pos)
    return self._initial_pos

  @initial_pos.setter
  def initial_pos(self, value: Any or np.ndarray or List[float] or Dict or float):
    """Set the initial position (state) of the environment
    """
    if type(value) == type(float):
      # Set default values
      for joint_idx, jt_name in enumerate(self.joints_to_control):
        # Prismatic joint is set to different value to ensure at least some part of robot tip
        # goes past the cannula
        if joint_idx == 2:
          self.obj_handle.set_joint_pos(jt_name, self.joint_limits['lower_limit'][2])
        else:
          self.obj_handle.set_joint_pos(jt_name, value)
      time.sleep(0.5)
    else:
      self._initial_pos = value
    return

# Overriding ARLEnv functions

  def reset(self) -> np.ndarray or List[float] or Dict:
    """Reset the robot environment

    Type 1 Reset : Uses the previous reached state as the initial state for
    next iteration

    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    observation, _, _, _ = self.step(action)

    Type 2 Reset : Sets the robot to a predefined initial state for each 
    iteration.
    """

    # Set the initial Position of the PSM
    self.initial_pos = 0.0
    initial_joint_pos, initial_joint_vel = self.get_joint_states()
    end_effector_frame = compute_FK(initial_joint_pos)
    # Updates the observation to the initialized position
    self._update_observation(end_effector_frame, initial_joint_pos, initial_joint_vel)
    # Samples a goal
    self.goal = self._sample_goal(self.obs)

    return self.obs.state

  def step(
    self,
    action  #: np.ndarray or List[float]
  ) -> Tuple[np.ndarray or List[float] or Dict,
             float,
             bool,
             Dict[str,
                  bool]]:
    """Performs the update step for the algorithm and dynamics
    """
    # Set incoming action to Action member variable
    self.action.action = action

    # Get the current state
    cur_state = copy.deepcopy(self.obs.state)

    # Make sure the action is valid
    self.action.check_if_valid_action()

    # counter for printing position, action, and reward
    self.count_for_print += 1

    # Compute the End Effector frame from the current joint states
    cur_joint_pos, cur_joint_vel = self.get_joint_states()
    cur_end_effector_frame = compute_FK(cur_joint_pos)

    # Get the positional component from end effector Homogenous Transform
    cur_end_effector_pos = np.asarray(cur_end_effector_frame[0:3, 3]).reshape(-1)
    # Apply the action and compute the resulting state
    next_end_effector_pos = self.action.apply(cur_end_effector_pos)
    # Clip the next state to ensure joint limits aren't broken
    next_end_effector_pos = self.check_if_valid_state(next_end_effector_pos)

    # Create a frame and maintain previous orientation
    next_end_effector_frame = cur_end_effector_frame
    for i in range(3):
      next_end_effector_frame[i, 3] = next_end_effector_pos[i]

    # Create a Homogenous Transform Message
    msg = HomogenousTransform()
    msg.data = np.array(next_end_effector_frame).flatten()

    rospy.wait_for_service('compute_IK')
    computed_joint_state = None
    try:
      compute_IK_service = rospy.ServiceProxy('compute_IK', ComputeIK)
      compute_IK_response = compute_IK_service.call(ComputeIKRequest(msg))
      computed_joint_state = list(compute_IK_response.q_des)
      for i in range(0, 6):
        computed_joint_state[i] = round(computed_joint_state[i], 4)
    except rospy.ServiceException as e:
      print("Service call failed: %s" % e, file=sys.stderr)
    # Ensure the computed joint positions are within the limit of user set joint positions
    next_joint_state = self.check_if_valid_joint_state(computed_joint_state)
    # Ensures that PSM joints reach the desired joint positions
    self.send_cmd(cmd=next_joint_state)

    # Update state, reward, done, and world values in the code
    self._update_observation(next_end_effector_frame, next_joint_state, cur_joint_vel)
    # Update the world handle
    self.world_handle.update()

    # Print function for viewing the output intermittently
    if self.count_for_print % self.steps_to_print == 0:
      print("Count: {} Goal: {}".format(self.count_for_print, self.goal.goal))
      print("\tState: {}".format(cur_state))
      print("\tAction: {}".format(self.action.action))
      print("\tReward: {}".format(self.obs.reward))

    return self.obs.state, self.obs.reward, self.obs.is_done, self.obs.info

  def send_cmd(self, cmd: Any or np.ndarray or List[float]):
    """Ensure the robot tip reaches the desired goal position before moving on to next iteration
    """

    count_for_joint_pos = 0
    while True:
      # Command joints to reach position
      for joint_idx, joint_name in enumerate(self.joints_to_control):
        self.obj_handle.set_joint_pos(joint_name, cmd[joint_idx])

      reached_joint_pos = np.zeros(7)
      # Check to see if desired joint positions have been reached
      for joint_idx, joint_name in enumerate(self.joints_to_control):
        reached_joint_pos[joint_idx] = self.obj_handle.get_joint_pos(joint_name)

      # Compare the error between desired and reached pos and allow acceptable margin
      error = np.around(np.subtract(cmd, reached_joint_pos), decimals=3)
      # Since Prismatic joint limits are smaller compared to the limits of other joints
      error[2] = np.around(np.subtract(cmd[2], reached_joint_pos[2]), decimals=4)
      # Create error margin vector
      error_margin = np.array([self.position_error_threshold] * len(self.joints_to_control))
      error_margin[2] = 0.5 * self.position_error_threshold

      # Check to ensure the error margins have been reached
      if (np.all(np.abs(error) <= error_margin)) or count_for_joint_pos > 75:
        break

      # Increment counter for iterations of checks
      count_for_joint_pos += 1

    return

  @abstractmethod
  def compute_reward(self, reached_goal: Goal, desired_goal: Goal, info: Dict[str, bool]) -> float:
    """Function to compute reward received by the agent
    """
    reward = 0.0
    return reward

  @abstractmethod
  def _sample_goal(self, observation: Observation) -> Goal:
    """Function to samples new goal positions and ensures its within the workspace of PSM
    """
    goal = Goal(0.0, 0.0, None)
    return goal


# PSM functions, can be imitated for other robots

  @abstractmethod
  def _update_observation(
    self,
    end_effector_frame: Any or np.ndarray,
    joint_pos: Any or np.ndarray,
    joint_vel: Any or np.ndarray
  ):
    """Update the observation object in this class
    """
    return

  def check_if_valid_state(self, state: np.ndarray or List[float]) -> np.ndarray or List[float]:
    """Clips the state if it goes beyond the cartesian limits
    """
    clipped_state = np.zeros(3)
    cartesian_pos_lower_limit = self.workspace_limits['lower_limit']
    cartesian_pos_upper_limit = self.workspace_limits['upper_limit']
    for axis in range(3):
      clipped_state[axis] = np.clip(
        state[axis],
        cartesian_pos_lower_limit[axis],
        cartesian_pos_upper_limit[axis]
      )
    return clipped_state

  def check_if_valid_joint_state(self,
                                 state: Any or np.ndarray
                                 or List[float]) -> Any or np.ndarray or List[float]:
    """Limits the joint states if it goes beyond the joint limits
    """
    # dvrk_limits_low = np.array([-1.605, -0.93556, -0.002444, -3.0456, -3.0414, -3.0481, -3.0498])
    # dvrk_limits_high = np.array([1.5994, 0.94249, 0.24001, 3.0485, 3.0528, 3.0376, 3.0399])
    # Note: Joint 5 and 6, joint pos = 0, 0 is closed jaw and 0.5, 0.5 is open
    limit_joint_values = np.zeros(7)
    joint_lower_limit = self.joint_limits['lower_limit']
    joint_upper_limit = self.joint_limits['upper_limit']
    for joint_idx in range(len(state)):
      limit_joint_values[joint_idx] = np.clip(
        state[joint_idx],
        joint_lower_limit[joint_idx],
        joint_upper_limit[joint_idx]
      )

    return limit_joint_values

  def get_joint_states(self) -> Tuple[List[float], List[float]]:
    """Computes the joint position and velocities
    """
    joint_positions = np.zeros(7)
    joint_velocities = np.zeros(7)

    for joint_idx, joint_name in enumerate(self.joints_to_control):
      joint_positions[joint_idx] = self.obj_handle.get_joint_pos(joint_name)
      joint_velocities[joint_idx] = self.obj_handle.get_joint_vel(joint_name)

    return joint_positions, joint_velocities

if __name__ == "__main__":
  # Create object of this class
  root_link = 'psm/baselink'

  # 'joints_to_control':
  #   np.array(
  #     [
  #       'baselink-yawlink',
  #       'yawlink-pitchbacklink',
  #       'pitchendlink-maininsertionlink',
  #       'maininsertionlink-toolrolllink',
  #       'toolrolllink-toolpitchlink',
  #       'toolpitchlink-toolgripper1link',
  #       'toolpitchlink-toolgripper2link'
  #     ]
  #   ),
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
  psmEnv = PSMCartesianDDPGEnv(**env_kwargs)
  psmEnv.make(root_link)
  # psmEnv.world_handle = psmEnv.ambf_client.get_world_handle()
  # psmEnv.world_handle.enable_throttling(False)