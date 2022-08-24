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

"""Test policy by playing a full Atari game."""

import itertools
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

import agent
import env_utils
import flax
import numpy as np
from dollar_lambda import argument, command
from flax.training import checkpoints
from lib import build_model, compute_step_values, get_initial_state
from main import GRAPHQL_ENDPOINT
from run_logger import RunLogger, get_load_params


def policy_test(
    apply_fn: Callable[..., Any],
    env_id: str,
    num_test_episodes: int,
    params: flax.core.frozen_dict.FrozenDict,
    render: bool,
    seed: int,
) -> np.float32:
    """
    Perform a test of the policy in Atari environment.

    Args:
      n_episodes: number of full Atari episodes to test on
      apply_fn: the actor-critic apply function
      params: actor-critic model parameters, they define the policy being tested
      game: defines the Atari game to test on

    Returns:
      total_reward: obtained score
    """
    test_env = env_utils.create_env(env_id=env_id, test=True)
    returns: list[float] = []
    for i in range(num_test_episodes):
        obs = test_env.reset(seed=seed + i)
        if render:
            test_env.render(mode="human")
        state = obs[None, ...]  # type: ignore # add batch dimension
        ep_return = 0.0
        for t in itertools.count():
            log_probs, _ = agent.policy_action(apply_fn, params, state)
            probs = np.exp(np.array(log_probs, dtype=np.float32))
            probabilities = probs[0] / probs[0].sum()
            action = test_env.np_random.choice(probs.shape[1], p=probabilities)
            obs, reward, done, _ = test_env.step(action)  # type: ignore
            if render:
                test_env.render(mode="human")
            ep_return += reward
            next_state = obs[None, ...] if not done else None
            state = next_state
            if done:
                returns += [ep_return]
                break
    return np.mean(returns)


def run(
    actor_steps: int,
    batch_size: int,
    decaying_lr_and_clip_param: bool,
    env_id: str,
    learning_rate: float,
    load_dir: Path,
    num_agents: int,
    num_epochs: int,
    seed: int,
    total_frames: int,
    **_
):
    loop_steps, iterations_per_step = compute_step_values(
        actor_steps=actor_steps,
        batch_size=batch_size,
        num_agents=num_agents,
        total_frames=total_frames,
    )
    env = env_utils.create_env(env_id, test=False)
    model = build_model(env_id)

    obs_shape = env.observation_space.shape
    assert obs_shape is not None
    state = get_initial_state(
        decaying_lr_and_clip_param=decaying_lr_and_clip_param,
        iterations_per_step=iterations_per_step,
        learning_rate=learning_rate,
        loop_steps=loop_steps,
        model=model,
        num_epochs=num_epochs,
        obs_shape=obs_shape,
        seed=seed,
    )
    logging.info("Loading model from %s", load_dir)
    state = checkpoints.restore_checkpoint(load_dir, state)
    return policy_test(
        apply_fn=state.apply_fn,
        env_id=env_id,
        num_test_episodes=100,
        params=state.params,
        render=True,
        seed=seed,
    )


@command(parsers=dict(run_id=argument("run_id", type=int)))
def main(run_id: int, load_dir: Optional[Path] = None):
    assert GRAPHQL_ENDPOINT is not None
    logger = RunLogger(GRAPHQL_ENDPOINT)
    params = get_load_params(run_id, logger)
    if load_dir is None:
        load_dir_str = os.getenv("SAVE_DIR")
        assert load_dir_str is not None
        load_dir = Path(load_dir_str)
    load_dir = Path(load_dir, str(run_id))
    return run(load_dir=load_dir, **params)


if __name__ == "__main__":
    main()
