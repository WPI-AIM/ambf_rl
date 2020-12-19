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

from typing import List, Set, Tuple, Dict, Any, Type
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
from abc import ABC, abstractmethod, ABCMeta

# from psmFK import *
# from transformations import euler_from_matrix
# from dvrk_functions.msg import HomogenousTransform
# import rospy
# from dvrk_functions.srv import *


# TODO: What makes an env suited for ddpg different from an env suited for HER
class Observation:

  def __init__(
    self,
    state: Any or np.ndarray or List[float] or Dict,
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

  def cur_observation(self) -> Tuple[Any or np.ndarray, float, bool, Dict]:
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

    self.action_space = spaces.Space()

    return

  def check_if_valid_action(self):
    """Check if the length of the action matches the defined action in the action space
    
    Raises
    ------
    TypeError
        If the length of action does not match the number of actions defined
    """

    assert len(self.action) == self.n_actions, TypeError("Incorrect length of actions provided!")
    # Clips the action value between the upper and lower limits
    np.clip(self.action, self.action_lims_low, self.action_lims_high, out=self.action)

    return

  def apply(self, state: np.ndarray, out: np.ndarray = None) -> np.ndarray:
    """Apply the action to the given state.

    This function can be overrided if the application of the action to state corresponds
    to a complex relationship to obtain the next state.

    Parameter
    ---------
    state : np.ndarray
        The state (s) to which the action should be added
    
    out : np.ndarray, optional
        The output parameter, passed by reference if the return argument is not used

    Raises
    ------
    Exception

    """
    if out is None:
      result = np.add(state, self.action)
    else:
      result = np.add(state, self.action, out=out)

    return result


class Goal:

  def __init__(
    self,
    goal_position_range: float,
    goal_error_margin: float,
    goal: Any or np.ndarray
  ) -> None:

    self.goal_position_range = goal_position_range
    self.goal_error_margin = goal_error_margin
    self.goal = goal

    return

  def _sample_goal(self) -> Any or np.ndarray:

    return self.goal


class ARLEnv(gym.Env, metaclass=ABCMeta):
  """Base class for the ARLEnv

  This class should not be instantiated on its own. It does not contain the 
  desired RL Algorithm or the Robot. This should remain an abstract class and
  should be wrapped by at least an algorithmic and robot environment wrapper.
  An example of this can be PSM_cartesian_env_ddpg(ARLEnv).

  Attributes
  ----------
  
  Methods
  -------
  skip_skipped_sim_steps(num)
      Define number of steps to skip if step-throttling is enabled.
  
  set_throttling_enable(check):
      Set the step-throttling Boolean.
  """

  def __init__(
    self,
    enable_step_throttling: bool,
    n_skip_steps: int,
    env_name: str = "arl_env"
  ) -> None:
    """Initialize the abstract class which handles all AMBF related interactions
    
    Parameters
    ----------
    enable_step_throttling : bool
        Flag to enable throttling of the simulator
    n_skip_steps : int
        Number of steps to skip after an update step
    env_name : str
        Name of the environment to train
    """
    super(ARLEnv, self).__init__()

    # Set the environment name
    self._env_name = env_name
    self._client_name = self._env_name + '_client'

    # Set environment and task parameters
    self._n_skip_steps = n_skip_steps
    self._enable_step_throttling = enable_step_throttling

    # Initialize sim steps
    self._skipped_sim_steps = 0
    self._prev_sim_step = 0
    self._count_for_print = 0

    # AMBF Sim Environment Declarations
    self._obj_handle = None  # Object
    self._world_handle = None  # World
    self._ambf_client = None  # Client(client_name=self._client_name)
    # Initialize the ambf client
    self.ambf_client = Client(client_name=self._client_name)

    # Set default Observation, Goal, and Action
    self._goal = Goal(0.0, 0.0, None)
    self._obs = Observation(None)
    self._action = Action(0, 0.0)

    # Set action space and observation space
    self.action_space = self.action.action_space
    self.observation_space = spaces.Space()

    # Sleep to allow the handle to connect to the AMBF server
    time.sleep(1)

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
  def skipped_sim_steps(self) -> int:
    """Return the simulation steps skipped
    """
    skipped_steps = 0
    if self.enable_step_throttling:
      while skipped_steps < self.n_skip_steps:
        skipped_steps = self.obj_handle.get_sim_step() - self.prev_sim_step
        time.sleep(1e-5)
      self.prev_sim_step = self.obj_handle.get_sim_step()
      if skipped_steps > self.n_skip_steps:
        print(
          'WARN: Skipped {} steps, Default skip limit {} Steps'.format(
            skipped_steps,
            self.n_skip_steps
          )
        )
    else:
      skipped_steps = self.obj_handle.get_sim_step() - self.prev_sim_step
      self.prev_sim_step = self.obj_handle.get_sim_step()

    self._skipped_sim_steps = skipped_steps

    return self._skipped_sim_steps

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
    self._obj_handle = value
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

  @property
  def action_space(self) -> spaces.Space:
    """Return the action_space
    """
    return self._action_space

  @action_space.setter
  def action_space(self, value: spaces.Space):
    """Set the action_space
    """
    self._action_space = value
    return

  @property
  def observation_space(self) -> spaces.Space:
    """Return the observation_space
    """
    return self._observation_space

  @observation_space.setter
  def observation_space(self, value: spaces.Space):
    """Set the observation_space
    """
    self._observation_space = value
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

    self.world_handle = self.ambf_client.get_world_handle()
    if self.world_handle is None:
      raise ClientHandleError("World handle not found, please make sure AMBF is running")
    self.obj_handle = self.ambf_client.get_obj_handle(robot_root_link)
    if self.obj_handle is None:
      raise ClientHandleError("Object handle not found, please make sure robot is loaded in AMBF")
    time.sleep(2)
    self.world_handle.enable_throttling(self.enable_step_throttling)
    self.world_handle.set_num_step_skips(self.n_skip_steps)

    # Verify obj_handle
    if self.obj_handle.get_num_joints() == 0:
      raise ClientHandleError(
        "Object handle returned {} objects, please make sure robot is loaded in AMBF".format(
          self.obj_handle.get_num_joints()
        )
      )

    return

  def seed(self, seed: int) -> List[int]:
    """Randomize the environment
    """
    self.np_random, seed = seeding.np_random(seed)

    return [seed]

  @abstractmethod
  def reset(self) -> np.ndarray or List[float] or Dict:
    """Reset the robot environment

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
  def step(
    self,
    action  #: Action
  ) -> Tuple[List[Any] or np.ndarray,
             float,
             bool,
             Dict[str,
                  bool]]:
    """Performs the update step for the algorithm and dynamics
    """
    return [], 0.0, False, {'': False}

  @abstractmethod
  def compute_reward(self, reached_goal: Goal, desired_goal: Goal, info: Dict[str, bool]) -> float:
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

  @abstractmethod
  def send_cmd(self, cmd: np.ndarray or List[float]):
    """Send the command to the robot in the AMBF Simulation.
    """
    return

if __name__ == '__main__':

  print(sys.path)