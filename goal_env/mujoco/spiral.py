# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility for loading the 2D navigation environments."""

import gym
import numpy as np

WALLS = {
    'Spiral':  # max_goal_dist = 45
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                  [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
                  [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                  [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
    'NineRooms':
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}


def resize_walls(walls, factor):
    (height, width) = walls.shape
    row_indices = np.array([i for i in range(height) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
    col_indices = np.array([i for i in range(width) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
    walls = walls[row_indices]
    walls = walls[:, col_indices]
    assert walls.shape == (factor * height, factor * width)
    return walls


class SpiralMazeEnv(gym.Env):
    """Abstract class for 2D navigation environments."""
    def __init__(self, walls = 'Spiral', resize_factor = 1):
        """Initialize the point environment.
        Args:
        walls: (str or array) binary, H x W array indicating locations of walls.
            Can also be the name of one of the maps defined above.
        resize_factor: (int) Scale the map by this factor.
        """
        if resize_factor > 1:
            self._walls = resize_walls(WALLS[walls], resize_factor)
        else:
            self._walls = WALLS[walls]
        self.resize_factor = resize_factor
        self.wall_name = walls
        (height, width) = self._walls.shape
        self._height = height
        self._width = width
        self._action_noise = 0.01
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float64)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([height, width, height, width]),
            dtype=np.float64)
        # self.reset()

    def _sample_empty_state(self):
        candidate_states = np.where(self._walls == 0)
        num_candidate_states = len(candidate_states[0])
        state_index = np.random.choice(num_candidate_states)
        state = np.array([candidate_states[0][state_index],
                        candidate_states[1][state_index]],
                        dtype=float)
        state += np.random.uniform(size=2)
        assert not self._is_blocked(state)
        return state

    def _get_obs(self):
        return self.state

    def sample_goal(self, fixed):
        if self.wall_name == 'Spiral':
            if fixed:
                return np.array([5., 5.], dtype=np.float64) * self.resize_factor
            else:
                return self._sample_empty_state()
        elif self.wall_name == 'NineRooms':
            if fixed:
                return np.array([15., 15.], dtype=np.float64) * self.resize_factor
            else:
                return self._sample_empty_state()
        else:
            raise ValueError('Unknown environment %s' % self.wall_name)
                
        
    def reset(self, fixed):
        if self.wall_name == 'Spiral':
            if fixed:
                self.state = np.array([9., 10.], dtype=np.float64) * self.resize_factor  
            else:
                self.state = self._sample_empty_state()
        elif self.wall_name == 'NineRooms':
            if fixed:
                self.state = np.array([1., 1.], dtype=np.float64) * self.resize_factor
            else:
                self.state = self._sample_empty_state()
        else:
            raise ValueError('Unknown environment %s' % self.wall_name)
        return self._get_obs()

    def _discretize_state(self, state, resolution=1.0):
        ij = np.floor(resolution * state).astype(int)
        ij = np.clip(ij, np.zeros(2), np.array(self.walls.shape) - 1)
        return ij.astype(int)

    def _is_blocked(self, state):
        assert len(state) == 2
        if (np.any(state < self.observation_space.low[:2])
            or np.any(state > self.observation_space.high[:2])):
            return True
        (i, j) = self._discretize_state(state)
        return (self._walls[i, j] == 1)

    def step(self, action):
        action = action.copy()
        if not self.action_space.contains(action):
            print('WARNING: clipping invalid action:', action)
        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise, (2,))
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)
        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = self.state.copy()
                new_state[axis] += dt * action[axis]
                if not self._is_blocked(new_state):
                    self.state = new_state

        obs = self._get_obs()
        done = False
        # dist = np.linalg.norm(self.goal - self.state)
        # rew = float(dist < 2.0)
        
        return obs, None, done, {}
          
    @property
    def walls(self):
        return self._walls

