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

"""Sequence Tagging example.

This script trains a Transformer on the Universal dependency dataset.
"""


import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import common_utils


def create_learning_rate_scheduler(
    factors="constant * linear_warmup * rsqrt_decay",
    base_learning_rate=0.5,
    warmup_steps=8000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000,
):
    """creates learning rate schedule.

    Interprets factors in the factors string which can consist of:
    * constant: interpreted as the constant value,
    * linear_warmup: interpreted as linear warmup until warmup_steps,
    * rsqrt_decay: divide by square root of max(step, warmup_steps)
    * decay_every: Every k steps decay the learning rate by decay_factor.
    * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

    Args:
      factors: a string with factors separated by '*' that defines the schedule.
      base_learning_rate: float, the starting constant for the lr schedule.
      warmup_steps: how many steps to warm up for in the warmup schedule.
      decay_factor: The amount to decay the learning rate by.
      steps_per_decay: How often to decay the learning rate.
      steps_per_cycle: Steps per cycle when using cosine decay.

    Returns:
      a function learning_rate(step): float -> {'learning_rate': float}, the
      step-dependent lr.
    """
    factors = [n.strip() for n in factors.split("*")]

    def step_fn(step):
        """Step to learning rate function."""
        ret = 1.0
        for name in factors:
            if name == "constant":
                ret *= base_learning_rate
            elif name == "linear_warmup":
                ret *= jnp.minimum(1.0, step / warmup_steps)
            elif name == "rsqrt_decay":
                ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
            elif name == "rsqrt_normalized_decay":
                ret *= jnp.sqrt(warmup_steps)
                ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
            elif name == "decay_every":
                ret *= decay_factor ** (step // steps_per_decay)
            elif name == "cosine_decay":
                progress = jnp.maximum(
                    0.0, (step - warmup_steps) / float(steps_per_cycle)
                )
                ret *= jnp.maximum(
                    0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0)))
                )
            else:
                raise ValueError("Unknown factor %s." % name)
        return jnp.asarray(ret, dtype=jnp.float32)

    return step_fn


def compute_cross_entropy(logits, targets):
    """Compute weighted cross entropy and entropy for log probs and targets.

    Args:
     logits: [batch, length, num_classes] float array.
     targets: categorical targets [batch, length] int array.
     weights: None or array of shape [batch x length]

    Returns:
      Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets"
            % (str(logits.shape), str(targets.shape))
        )
    onehot_targets = common_utils.onehot(targets, logits.shape[-1])
    loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
    return loss.mean()


def compute_accuracy(logits, targets):
    """Compute weighted accuracy for log probs and targets.

    Args:
     logits: [batch, length, num_classes] float array.
     targets: categorical targets [batch, length] int array.
     weights: None or array of shape [batch x length]

    Returns:
      Tuple of scalar accuracy and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets"
            % (str(logits.shape), str(targets.shape))
        )
    loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
    return loss.mean()


def compute_metrics(logits, labels):
    """Compute summary metrics."""
    loss = compute_cross_entropy(logits, labels)
    acc = compute_accuracy(logits, labels)
    metrics = {"loss": loss, "accuracy": acc}
    metrics = np.sum(metrics, -1)  # type: ignore
    return metrics


def compute_loss(estimate, targets):
    return jnp.mean(jnp.square(estimate - targets))


def compute_error(estimate, targets):
    return jnp.mean(jnp.abs(estimate - targets))


def eval_step(params, batch, model):
    """Calculate evaluation metrics on a batch."""
    output = model.apply({"params": params}, inputs=batch, train=False)
    targets = get_targets(batch)
    return {
        "loss": compute_cross_entropy(output, targets),
        "accuracy": compute_accuracy(output, targets),
    }


def get_targets(batch):
    return batch["action"]


def train_step(state, batch, model, learning_rate_fn, dropout_rng=None):
    """Perform a single training step."""
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        """loss function used for training."""
        output = model.apply(
            {"params": params},
            inputs=batch,
            train=True,
            rngs={"dropout": dropout_rng},
        )
        targets = get_targets(batch)
        return compute_cross_entropy(output, targets), compute_accuracy(output, targets)

    lr = learning_rate_fn(state.step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, acc), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = {"accuracy": acc, "learning rate": lr, "loss": loss}
    return new_state, metrics


def pad_examples(x, desired_batch_size):
    """Expand batch to desired size by zeros with the shape of last slice."""
    batch_pad = desired_batch_size - x.shape[0]
    # Padding with zeros to avoid that they get counted in compute_metrics.
    return np.concatenate([x, np.tile(np.zeros_like(x[-1]), (batch_pad, 1))])
