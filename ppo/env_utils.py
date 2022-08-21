# Copyright 2022 The Flax Authors.
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

"""Utilities for handling the Atari environment."""


import operator
from functools import reduce

import gym
import numpy as np
from gym import RewardWrapper  # type: ignore
from gym.core import ObservationWrapper
from gym.spaces import Dict, Discrete, MultiBinary, MultiDiscrete
from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace
from rich.pretty import pprint
from rich.text import Text


def join_text(*text: Text, joiner: str) -> Text:
    head, *tail = text
    return reduce(operator.add, [head] + [Text(joiner) + t for t in tail])


class ObsGoalWrapper(ObservationWrapper):
    def __init__(self, env: "EmptyEnv"):
        super().__init__(env)

        coord_space = MultiDiscrete(np.array([env.width, env.height]))
        assert isinstance(self.observation_space, Dict)
        self.observation_space = Dict(
            dict(**self.observation_space.spaces, agent=coord_space, goal=coord_space)
        )

    def observation(self, obs):
        assert isinstance(self.env, EmptyEnv)
        return dict(**obs, agent=self.env.agent_pos, goal=self.env.goal_pos)


class FlatObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(self.observation_space, Dict)
        spaces = self.observation_space.spaces
        agent_space = spaces["agent"]
        dir_space = spaces["direction"]
        goal_space = spaces["goal"]
        assert isinstance(agent_space, MultiDiscrete)
        assert isinstance(dir_space, Discrete)
        assert isinstance(goal_space, MultiDiscrete)
        self.observation_space = MultiDiscrete(
            np.array([*agent_space.nvec, dir_space.n, *goal_space.nvec])
        )

    def observation(self, obs):
        return np.concatenate([obs["agent"], np.array([obs["direction"]]), obs["goal"]])


class OneHotWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        space = self.observation_space
        self.observation_space = gym.spaces.MultiBinary(
            np.array([*space.nvec.shape, space.nvec.max()])
        )
        self.one_hot = np.eye(space.nvec.max(), dtype=np.int)

    def observation(self, obs):
        return self.one_hot[obs]


class FlattenWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(self.observation_space, MultiBinary)
        self.observation_space = MultiBinary(int(np.prod(self.observation_space.n)))

    def observation(self, obs):
        return obs.flatten()


class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(
            mission_func=lambda: "get to the green goal square"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs,
        )

    @staticmethod
    def ascii_of_image(image: np.ndarray):
        def rows():
            for row in image:
                yield join_text(
                    *[
                        Text("██", style=f"rgb({','.join(rgb.astype(int))})")
                        for rgb in row
                    ],
                    joiner="",
                )

        return join_text(*rows(), joiner="\n")

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.goal_pos = self.place_obj(Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def reset(self):
        self.__action = self.__reward = self.__done = None
        return super().reset()

    def step(self, action):
        self.__action = action
        state, self.__reward, self.__done, info = super().step(action)
        return state, self.__reward, self.__done, info

    def render(self, mode="human", highlight=True, tile_size=...):
        print(self.ascii_of_image(self.render_obs()))
        print()
        subtitle = ""
        if self.__action is not None:
            subtitle += f", {self.__action.name.replace('_', ' ')}"
        if self._reward is not None:
            assert isinstance(self.__reward, float)
            subtitle += f", r={round(self.__reward, 2)}"
        if self.__done:
            subtitle += ", done"
        pprint(subtitle.swapcase())
        input("Press enter to continue.")


class ClipRewardEnv(RewardWrapper):
    """Adapted from OpenAI baselines.

    github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """

    def __init__(self, env):
        RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


def create_env():
    """Create a FrameStack object that serves as environment for the `game`."""
    env = EmptyEnv(agent_start_pos=None)
    env = ObsGoalWrapper(env)
    env = FlatObsWrapper(env)
    env = OneHotWrapper(env)
    env = FlattenWrapper(env)
    return env


def get_num_actions(game: str):
    """Get the number of possible actions of a given Atari game.

    This determines the number of outputs in the actor part of the
    actor-critic model.
    """
    env = gym.make(game)
    assert isinstance(env.action_space, Discrete)
    return env.action_space.n
