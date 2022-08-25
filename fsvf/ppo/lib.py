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

"""Library file which executes the PPO training."""

import functools
import itertools
import logging
import os
import re
import time
from dataclasses import asdict, astuple
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import optax
from flax import linen as nn
from flax.training import checkpoints, train_state
from gym_minigrid.minigrid import MiniGridEnv
from jax.tree_util import tree_map
from ppo import env_utils
from ppo.agent import ExpTuple, RemoteSimulator, policy_action
from ppo.models import OneHotConv, RGBConv, TwoLayer
from returns.curry import partial
from returns.pipeline import flow
from rich.console import Console
from run_logger import RunLogger

GRAPHQL_ENDPOINT = os.getenv("GRAPHQL_ENDPOINT")


@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(
    rewards: np.ndarray,
    terminal_masks: np.ndarray,
    values: np.ndarray,
    discount: float,
    gae_param: float,
):
    """Use Generalized Advantage Estimation (GAE) to compute advantages.

    As defined by eqs. (11-12) in PPO paper arXiv: 1707.06347. Implementation uses
    key observation that A_{t} = delta_t + gamma*lambda*A_{t+1}.

    Args:
      rewards: array shaped (actor_steps, num_agents), rewards from the game
      terminal_masks: array shaped (actor_steps, num_agents), zeros for terminal
                      and ones for non-terminal states
      values: array shaped (actor_steps, num_agents), values estimated by critic
      discount: RL discount usually denoted with gamma
      gae_param: GAE parameter usually denoted with lambda

    Returns:
      advantages: calculated advantages shaped (actor_steps, num_agents)
    """
    assert rewards.shape[0] + 1 == values.shape[0], (
        "One more value needed; Eq. "
        "(12) in PPO paper requires "
        "V(s_{t+1}) for delta_t"
    )
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        # Masks used to set next state value to 0 for terminal states.
        value_diff = discount * values[t + 1] * terminal_masks[t] - values[t]
        delta = rewards[t] + value_diff
        # Masks[t] used to ensure that values before and after a terminal state
        # are independent of each other.
        gae = delta + discount * gae_param * terminal_masks[t] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return jnp.array(advantages)


def loss_fn(
    params: flax.core.FrozenDict,
    apply_fn: Callable[..., Any],
    minibatch: Tuple,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
):
    """Evaluate the loss function.

    Compute loss as a sum of three components: the negative of the PPO clipped
    surrogate objective, the value function loss and the negative of the entropy
    bonus.

    Args:
      params: the parameters of the actor-critic model
      apply_fn: the actor-critic model's apply function
      minibatch: Tuple of five elements forming one experience batch:
                 states: shape (batch_size, 84, 84, 4)
                 actions: shape (batch_size, 84, 84, 4)
                 old_log_probs: shape (batch_size,)
                 returns: shape (batch_size,)
                 advantages: shape (batch_size,)
      clip_param: the PPO clipping parameter used to clamp ratios in loss function
      vf_coeff: weighs value function loss in total loss
      entropy_coeff: weighs entropy bonus in the total loss

    Returns:
      loss: the PPO loss, scalar quantity
    """
    states, actions, old_log_probs, returns, advantages = minibatch
    log_probs, values = policy_action(apply_fn, params, states)
    values = values[:, 0]  # Convert shapes: (batch, 1) to (batch, ).
    probs = jnp.exp(log_probs)

    value_loss = jnp.mean(jnp.square(returns - values), axis=0)

    entropy = jnp.sum(-probs * log_probs, axis=1).mean()

    log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
    ratios = jnp.exp(log_probs_act_taken - old_log_probs)
    # Advantage normalization (following the OpenAI baselines).
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(
        1.0 - clip_param, ratios, 1.0 + clip_param
    )
    ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss), axis=0)

    return ppo_loss + vf_coeff * value_loss - entropy_coeff * entropy


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(
    state: train_state.TrainState,
    trajectories: Tuple,
    batch_size: int,
    *,
    clip_param: float,
    vf_coeff: float,
    entropy_coeff: float,
):
    """Compilable train step.

    Runs an entire epoch of training (i.e. the loop over minibatches within
    an epoch is included here for performance reasons).

    Args:
      state: the train state
      trajectories: Tuple of the following five elements forming the experience:
                    states: shape (steps_per_agent*num_agents, 84, 84, 4)
                    actions: shape (steps_per_agent*num_agents, 84, 84, 4)
                    old_log_probs: shape (steps_per_agent*num_agents, )
                    returns: shape (steps_per_agent*num_agents, )
                    advantages: (steps_per_agent*num_agents, )
      batch_size: the minibatch size, static argument
      clip_param: the PPO clipping parameter used to clamp ratios in loss function
      vf_coeff: weighs value function loss in total loss
      entropy_coeff: weighs entropy bonus in the total loss

    Returns:
      optimizer: new optimizer after the parameters update
      loss: loss summed over training steps
    """
    iterations = trajectories[0].shape[0] // batch_size
    trajectories = tree_map(
        lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), trajectories
    )
    loss = 0.0
    for batch in zip(*trajectories):
        grad_fn = jax.value_and_grad(loss_fn)
        l, grads = grad_fn(
            state.params, state.apply_fn, batch, clip_param, vf_coeff, entropy_coeff
        )
        loss += l
        state = state.apply_gradients(grads=grads)
    return state, loss


def get_experience(
    rng: np.random.Generator,
    simulators: List[RemoteSimulator],
    state: train_state.TrainState,
):
    """Collect experience from agents.

    Runs `steps_per_actor` time steps of the game for each of the `simulators`.
    """
    # Range up to steps_per_actor + 1 to get one more value needed for GAE.
    while True:
        sim_states = []
        for sim in simulators:
            sim_state = sim.conn.recv()
            sim_states.append(sim_state)
        sim_states = np.concatenate(sim_states, axis=0)
        log_probs, values = policy_action(state.apply_fn, state.params, sim_states)
        log_probs, values = jax.device_get((log_probs, values))
        probs = np.exp(np.array(log_probs))
        for i, sim in enumerate(simulators):
            probabilities = probs[i]
            action = rng.choice(probs.shape[1], p=probabilities)
            sim.conn.send(action)
        experiences = []
        for i, sim in enumerate(simulators):
            sim_state, action, reward, done = sim.conn.recv()
            value = values[i, 0]
            log_prob = log_probs[i][action]
            sample = ExpTuple(sim_state, action, reward, value, log_prob, done)
            experiences.append(sample)
        yield experiences


def process_experience(
    experience: List[List[ExpTuple]],
    actor_steps: int,
    num_agents: int,
    gamma: float,
    lambda_: float,
    obs_shape: List[int],
):
    """Process experience for training, including advantage estimation.

    Args:
      experience: collected from agents in the form of nested lists/namedtuple
      actor_steps: number of steps each agent has completed
      num_agents: number of agents that collected experience
      gamma: dicount parameter
      lambda_: GAE parameter

    Returns:
      trajectories: trajectories readily accessible for `train_step()` function
    """
    exp_dims = (actor_steps, num_agents)
    values_dims = (actor_steps + 1, num_agents)
    states = np.zeros([*exp_dims, *obs_shape], dtype=np.float32)
    actions = np.zeros(exp_dims, dtype=np.int32)
    rewards = np.zeros(exp_dims, dtype=np.float32)
    values = np.zeros(values_dims, dtype=np.float32)
    log_probs = np.zeros(exp_dims, dtype=np.float32)
    dones = np.zeros(exp_dims, dtype=np.float32)

    for t in range(len(experience) - 1):  # experience[-1] only for next_values
        for agent_id, exp_agent in enumerate(experience[t]):
            states[t, agent_id, ...] = exp_agent.state
            actions[t, agent_id] = exp_agent.action
            rewards[t, agent_id] = exp_agent.reward
            values[t, agent_id] = exp_agent.value
            log_probs[t, agent_id] = exp_agent.log_prob
            # Dones need to be 0 for terminal states.
            dones[t, agent_id] = float(not exp_agent.done)
    for a in range(num_agents):
        values[-1, a] = experience[-1][a].value
    advantages = gae_advantages(rewards, dones, values, gamma, lambda_)
    returns = advantages + values[:-1, :]
    # After preprocessing, concatenate data from all agents.
    trajectories = (states, actions, log_probs, returns, advantages)
    trajectory_len = num_agents * actor_steps
    trajectories = tuple(  # type: ignore
        map(lambda x: np.reshape(x, (trajectory_len,) + x.shape[2:]), trajectories)
    )
    return trajectories


@functools.partial(jax.jit, static_argnums=[0, 2])
def get_initial_params(input_dims: List[int], key: np.ndarray, model: nn.Module):
    input_dims = [1, *input_dims]  # (minibatch, height, width, stacked frames)
    init_shape = jnp.ones(input_dims, jnp.float32)
    initial_params = model.init(key, init_shape)["params"]
    return initial_params


def create_train_state(
    decaying_lr_and_clip_param: bool,
    params,
    learning_rate: float,
    model: nn.Module,
    train_steps: int,
) -> train_state.TrainState:
    if decaying_lr_and_clip_param:
        lr = optax.linear_schedule(
            init_value=learning_rate, end_value=0.0, transition_steps=train_steps
        )
    else:
        lr = learning_rate
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


def train(
    # Number of steps each agent performs in one policy unroll.
    actor_steps: int,
    # Batch size used in training.
    batch_size: int,
    # The PPO clipping parameter used to clamp ratios in loss function.
    clip_param: float,
    # Linearly decay learning rate and clipping parameter to zero during
    # the training.
    decaying_lr_and_clip_param: bool,
    # Weight of entropy bonus in the total loss.
    entropy_coeff: float,
    # id to pass to gym.make
    env_id: str,
    # If not none, path to save experience from test rollouts.
    experience_dir: Optional[str],
    # RL discount parameter.
    gamma: float,
    # Generalized Advantage Estimation parameter.
    lambda_: float,
    # The learning rate for the Adam optimizer.
    learning_rate: float,
    # If not none, parameters will be loaded from this path.
    load_dir: Optional[str],
    # number of updates between logs
    log_frequency: int,
    # logger for logging to Hasura
    logger: RunLogger,
    # Number of agents playing in parallel.
    num_agents: int,
    # Number of training epochs per each unroll of the policy.
    num_epochs: int,
    # Number of agents to run tests on.
    num_test_agents: int,
    # Number of episodes to run test agents for.
    num_test_episodes: int,
    # whether to render during testing
    render: bool,
    # directory to save model checkpoints in.
    save_dir: str,
    # number of updates between checkpoints.
    save_frequency: int,
    # random seed
    seed: int,
    # Total number of frames seen during training.
    total_frames: int,
    # Weight of value function loss in the total loss.
    vf_coeff: float,
):
    """Main training loop.

    Args:
      model: the actor-critic model
      config: object holding hyperparameters and the training information
      model_dir: path to dictionary where checkpoints and logging info are stored

    Returns:
      optimizer: the trained optimizer
    """
    console = Console()
    if render:
        num_agents = 1

    def get_simulators(n: int):
        return [RemoteSimulator(env_id=env_id, seed=seed + i) for i in range(n)]

    simulators = get_simulators(num_agents)
    test_simulators = get_simulators(num_test_agents)

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
    rng = np.random.default_rng(seed)
    test_rng = np.random.default_rng(seed)
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
    if load_dir is not None:
        logging.info("Loading model from %s", load_dir)
        state = checkpoints.restore_checkpoint(load_dir, state)

    save_count = 0
    start_step = int(state.step) // num_epochs // iterations_per_step
    start_time = time.time()
    console.log(f"Start training from step: {start_step}")

    for step in range(start_step, loop_steps):
        # Bookkeeping and testing.
        if step % log_frequency == 0:
            experiences = []
            returns = 0.0
            episodes = 0
            for exp in get_experience(
                rng=test_rng, simulators=test_simulators, state=state
            ):
                experiences += [exp]
                returns += sum([e.reward for e in exp])
                episodes += sum([e.done for e in exp])
                if episodes >= num_test_episodes:
                    break

            test_return = returns / episodes

            experience_dir = experience_dir or os.getenv("EXPERIENCE_DIR")
            if experience_dir is not None:
                i = 0
                for episode in flow(
                    zip(*experiences), list, partial(tree_map, astuple)
                ):
                    unclipped = flow(
                        zip(*episode),
                        partial(map, np.stack),
                        lambda s: ExpTuple(*s),
                    )
                    while unclipped.done.any():
                        [[first_term_step, *_]] = unclipped.done.nonzero()
                        first_term_step += 1

                        def clip(cut_fn):
                            return flow(
                                unclipped,
                                astuple,
                                partial(map, cut_fn),
                                lambda s: ExpTuple(*s),
                            )

                        clipped = clip(lambda s: s[:first_term_step])
                        assert clipped.done[-1]
                        unclipped = clip(lambda s: s[first_term_step:])
                        step_dir = Path(experience_dir, str(step))
                        step_dir.mkdir(parents=True, exist_ok=True)

                        with Path(step_dir, f"{i}.npz").open("wb") as f:
                            np.savez(f, **asdict(clipped))
                            console.log(f"Saved experience to {f.name}")
                            i += 1
            frames = step * num_agents * actor_steps
            log = dict(step=frames, hours=(time.time() - start_time) / 3600) | {
                "return": test_return,
                "run ID": logger.run_id,
                "save count": save_count,
            }
            console.log(log)
            if logger.run_id is not None:
                logger.log(**log)

        # Core training code.
        alpha = 1.0 - step / loop_steps if decaying_lr_and_clip_param else 1.0
        all_experiences = flow(
            get_experience(rng=rng, simulators=simulators, state=state),
            lambda exp: itertools.islice(exp, actor_steps + 1),
            list,
        )
        trajectories = process_experience(
            actor_steps=actor_steps,
            experience=all_experiences,
            gamma=gamma,
            lambda_=lambda_,
            num_agents=num_agents,
            obs_shape=list(obs_shape),
        )
        clip_param = clip_param * alpha
        for _ in range(num_epochs):
            permutation = rng.permutation(num_agents * actor_steps)
            trajectories = tuple(x[permutation] for x in trajectories)
            state, _ = train_step(
                state,
                trajectories,
                batch_size,
                clip_param=clip_param,
                vf_coeff=vf_coeff,
                entropy_coeff=entropy_coeff,
            )
        if save_frequency and ((step + 1) % save_frequency == 0):
            _save_dir = save_dir or os.getenv("SAVE_DIR")
            assert _save_dir is not None
            checkpoints.save_checkpoint(
                Path(_save_dir, str(logger.run_id)),
                target=state,
                step=step + 1,
                overwrite=True,
            )
            save_count += 1
    return state


def get_initial_state(
    decaying_lr_and_clip_param,
    learning_rate,
    num_epochs,
    seed,
    loop_steps,
    iterations_per_step,
    model,
    obs_shape,
):
    key = jax.random.PRNGKey(seed)
    initial_params = get_initial_params(
        input_dims=obs_shape,
        key=key,
        model=model,
    )
    state = create_train_state(
        decaying_lr_and_clip_param=decaying_lr_and_clip_param,
        params=initial_params,
        learning_rate=learning_rate,
        model=model,
        train_steps=loop_steps * num_epochs * iterations_per_step,
    )
    del initial_params
    return state


def build_model(env_id):
    if env_id == "empty":
        num_actions = len(MiniGridEnv.Actions)
        model = RGBConv(num_outputs=num_actions)
    elif re.match(env_utils.MyEnv.pattern, env_id):
        num_actions = env_utils.MyEnv.action_space.n
        model = OneHotConv(num_outputs=num_actions)
    elif "NoFrameskip" in env_id:
        num_actions = env_utils.get_num_actions(env_id)
        model = RGBConv(num_outputs=num_actions)
    elif "MiniGrid" in env_id:
        num_actions = env_utils.get_num_actions(env_id)
        model = RGBConv(num_outputs=num_actions)
    else:
        num_actions = env_utils.get_num_actions(env_id)
        model = TwoLayer(num_outputs=num_actions)
    return model


def compute_step_values(actor_steps, batch_size, num_agents, total_frames):
    loop_steps = total_frames // (num_agents * actor_steps)
    # train_step does multiple steps per call for better performance
    # compute number of steps per call here to convert between the number of
    # train steps and the inner number of optimizer steps
    iterations_per_step = num_agents * actor_steps // batch_size
    return loop_steps, iterations_per_step
