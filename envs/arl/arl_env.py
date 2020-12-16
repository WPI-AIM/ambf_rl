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
import ambf_client
from gym import spaces
import numpy as np
import copy
import time
import gym
from gym.utils import seeding
from ambf_world import World
from ambf_object import Object
import sys
from abc import ABC, abstractmethod

# from psmFK import *
# from transformations import euler_from_matrix
# from dvrk_functions.msg import HomogenousTransform
# import rospy
# from dvrk_functions.srv import *


# TODO: What makes an env suited for ddpg different from an env suited for HER
class Observation:

  def __init__(
    self,
    state: Any,
    dist: int = 0,
    reward: float = 0.0,
    prev_reward: float = 0.0,
    cur_reward: float = 0.0,
    is_done: bool = False,
    info: Dict = {},
    sim_step_no: int = 0
  ) -> None:

    self.state = state
    self.dist = dist
    self.reward = reward
    self.prev_reward = prev_reward
    self.cur_reward = cur_reward
    self.is_done = is_done
    self.info = info
    self.sim_step_no = sim_step_no

    return

  def cur_observation(self) -> Tuple[Any, float, bool, Dict]:
    return self.state, self.reward, self.is_done, self.info


class ClientHandleError(BaseException):

  def __init__(self, *args: object) -> None:
    super().__init__(*args)


class Action:

  def __init__(
    self,
    n_actions: int,
    action_space_limit: float,
    action_lims_low: List[float] = None,
    action_lims_high: List[float] = None
  ) -> None:
    self.n_actions = n_actions
    self.action = [0.0 for i in range(n_actions)]
    self.action_space_limit = action_space_limit
    if action_lims_low is None:
      self.action_lims_low = -action_space_limit * np.ones(self.n_actions)
    else:
      self.action_lims_low = action_lims_low

    if action_lims_high is None:
      self.action_lims_high = action_space_limit * np.ones(self.n_actions)
    else:
      self.action_lims_high = action_lims_high

    self.action_space = spaces.Box(
      low=-action_space_limit,
      high=action_space_limit,
      shape=(self.n_actions,
             ),
      dtype="float32"
    )

    return

  def check_if_valid_action(self, action: List[float]) -> List[float]:

    assert len(action) == self.n_actions, TypeError("Incorrect length of actions provided!")
    self.action = np.clip(action, self.action_lims_low, self.action_lims_high)

    return self.action


class Goal:

  def __init__(self, goal_position_range: float, goal_error_margin: float, goal: Any) -> None:

    self.goal_position_range = goal_position_range
    self.goal_error_margin = goal_error_margin
    self.goal = goal

    return

  def _sample_goal(self) -> Any:

    return self.goal


class ARLEnv(gym.Env, ABC):
  """Base class for the ARLEnv

  This class should not be instantiated on its own. It does not contain the 
  desired RL Algorithm or the Robot. This should remain an abstract class and
  should be wrapped by at least an algorithmic and robot environment wrapper.
  An example of this can be PSM_cartesian_env_ddpg(ARLEnv).

  Attributes
  ----------
  
  Methods
  -------
  skip_sim_steps(num)
      Define number of steps to skip if step-throttling is enabled.
  
  set_throttling_enable(check):
      Set the step-throttling Boolean.
  """

  def __init__(
    self,
    enable_step_throttling:bool,
    n_skip_steps:int=5,           # should we pass these parameters as well?
    env_name:str="arl_env"
  ) -> None:
    """
    TODO: Ask vignesh - static types for parameters to help with documentation.
    
    Parameters
    ----------
    name : str
        The name of the animal
    sound : str
        The sound the animal makes
    num_legs : int, optional
        The number of legs the animal (default is 4)
    """
    super(ARLEnv, self).__init__()

    # Set the environment name
    self._env_name = env_name
    self._client_name = self._env_name + '_client'

    # Set environment and task parameters
    self._n_skip_steps = n_skip_steps
    self._enable_step_throttling = enable_step_throttling

    # Initialize sim steps
    self._prev_sim_step = 0
    self._count_for_print = 0

    # AMBF Sim Environment Declarations
    self._obj_handle = Object
    self._world_handle = World
    self._ambf_client = Client(client_name=self._client_name)
    # Initialize the ambf client
    self._ambf_client.connect()

    # Set default Observation, Goal, and Action
    self._goal = Goal(0.0, 0.0, None)
    self._obs = Observation(None)
    self._action = Action(0, 0.0)

    # Sleep to allow the handle to connect to the AMBF server
    time.sleep(1)
    

    # Populate all objects of the robot within a common namespace
    self._ambf_client.create_objs_from_rostopics()

    # Random seed the environment
    self.seed(5)

    return

# Properties

  @property
  def env_name(self) -> str:
    """Return the name of the environment
    """
    return self._env_name

  @env_name.setter
  def env_name(self, value: str):
    """Set the environment name
    """
    self._env_name = value

  @property
  def client_name(self) -> str:
    """Return the AMBF Client Name
    """
    return self._client_name

  @client_name.setter
  def client_name(self, value: str):
    """Sets the AMBF Client Name
    """
    self._client_name = value
    return

  @property
  def n_skip_steps(self) -> int:
    """Return number of steps to skip.

    TODO: Provide reference for step-throttling
    """
    return self._n_skip_steps

  @n_skip_steps.setter
  def n_skip_steps(self, value: int):
    """Define number of steps to skip if step-throttling is enabled.
    """
    self._n_skip_steps = value
    self._world_handle.set_num_step_skips(value)
    return

  @property
  def enable_step_throttling(self) -> bool:
    """Return the step-throttling state

    TODO: Provide reference for step-throttling
    """
    return self._enable_step_throttling

  @enable_step_throttling.setter
  def enable_step_throttling(self, value: bool):
    """Set the step-throttling state
    """
    self._enable_step_throttling = value
    self._world_handle.enable_throttling(self._enable_step_throttling)
    return

  @property
  def prev_sim_step(self) -> int:
    """Return the previous simulation step number
    """
    return self._prev_sim_step

  @prev_sim_step.setter
  def prev_sim_step(self, value: int):
    """Set the previous simulation step number
    """
    self._prev_sim_step = value
    return

  @property
  def count_for_print(self) -> int:
    """Return the number of counts to print
    """
    return self._count_for_print

  @count_for_print.setter
  def count_for_print(self, value: int):
    """Set the number of counts to print
    """
    self._count_for_print = value
    return

  @property
  def obj_handle(self) -> Object:
    """Return the AMBF Object Handle
    """
    return self._obj_handle

  @obj_handle.setter
  def obj_handle(self, value: Object):
    """Set the AMBF Object Handle
    """
    self._obj_handle = Object
    return

  @property
  def world_handle(self) -> World:
    """Return the AMBF World Handle
    """
    return self._world_handle

  @world_handle.setter
  def world_handle(self, value: World):
    """Set the AMBF World Handle
    """
    self._world_handle = value
    return

  @property
  def ambf_client(self) -> Client:
    """Return the AMBF Client
    """
    return self._ambf_client

  @ambf_client.setter
  def ambf_client(self, value: Client):
    """Set the AMBF Client
    """
    self._ambf_client = value
    self._ambf_client.connect()
    time.sleep(1)
    self._ambf_client.create_objs_from_rostopics()
    return

  @property
  def goal(self) -> Goal:
    """Return the goal object
    """
    return self._goal

  @goal.setter
  def goal(self, value: Goal):
    """Set the goal object
    """
    self._goal = value
    return

  @property
  def obs(self) -> Observation:
    """Return the Observation object
    """
    return self._obs

  @obs.setter
  def obs(self, value: Observation):
    """Set the Observation object
    """
    self._obs = value
    return

  @property
  def action(self) -> Action:
    """Return the action object
    """
    return self._action

  @action.setter
  def action(self, value: Action):
    """Set the action object
    """
    self._action = value
    return

# Gym requirements

  def make(self, robot_root_link: str):
    """Creates an object handle of the robot and world in AMBF

    Parameters
    ----------
    robot_root_link : string  
    
        Name of the root link of the robot.  

        eg. for the dVRK PSM: robot_root_link = 'psm/baselink'  

    Raises
    ------
    ClientHandleError
        If obj handle or world handle are None, then a ClientHandleError is
        raised.
    """

    self.obj_handle = self.ambf_client.get_obj_handle(robot_root_link)
    self.world_handle.enable_throttling(self.enable_step_throttling)
    self.world_handle.set_num_step_skips(self.n_skip_steps)
    time.sleep(2)
    if self.obj_handle is None or self.world_handle is None:
      raise ClientHandleError("Client handles not found")

    return

  def seed(self, seed: int) -> List[int]:
    """Randomize the environment
    """
    self.np_random, seed = seeding.np_random(seed)

    return [seed]

  @abstractmethod
  def reset(self) -> Observation:
    """Reset the robot environment

    Type 1 Reset : Uses the previous reached state as the initial state for
    next iteration

    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    observation, _, _, _ = self.step(action)

    Type 2 Reset : Sets the robot to a predefined initial state for each 
    iteration.

    Raises
    ------
    NotImplementedError
        If function is not overridden, a Not Implemented error is raised if the 
        reset function from the base class is called.
    """
    raise NotImplementedError("Abstract method reset needs to be overridden")
    return Observation(None)

  def render(self, mode):
    return

  @abstractmethod
  def step(self, action: Action) -> Tuple[List[Any] or Any, float, bool, Dict[str, bool]]:
    """Performs the update step for the algorithm and dynamics
    """
    return [], 0.0, False, {'': False}

  @abstractmethod
  def compute_reward(self, achieved_goal: Goal, goal: Goal, info: Dict[str, bool]) -> float:
    """Function to compute reward received by the agent
    """
    return 0.0

  @abstractmethod
  def _sample_goal(self, observation: Observation) -> Goal:
    """Function to samples new goal positions and ensures its within the workspace of PSM
    """
    return Goal(0.0, 0.0, None)

  def _check_if_done(self) -> bool:
    """Function to check if the episode was successful
    """
    if abs(self.obs.dist) < self.goal.goal_error_margin:
      return True
    else:
      return False

  # @abstractmethod
  def _update_info(self):
    """Can be used to Provide information for debugging purpose

    TODO: Should this function be made abstract?
    """
    info = {'is_success': self._is_success()}
    return info

  # @abstractmethod
  def _is_success(self):
    """Function to check if the robot reached the desired goal within a predefined error margin
    
    TODO: Should this function be made abstract?
    """
    return self._check_if_done()


# PSM and should be moved to inherited class

# def set_initial_pos_func(self):

#   return

# def get_joint_pos_vel_func(self) -> Tuple[str, str]:

#   return 'a', 'b'

# def compute_fk(self, states):

#   return None

# # Use to ensure that goal position is reached before next incoming goal
# def set_commanded_joint_pos(self):
#   return

# def limit_cartesian_pos(self):
#   return

# def limit_joint_pos(self):
#   return

# def _update_observation(self):
#   return

if __name__ == '__main__':

  print(sys.path)