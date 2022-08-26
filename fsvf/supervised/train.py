import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import supervised.dataset  # noqa: F401
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import jax_utils
from flax.training import common_utils, train_state
from jax import random
from returns.curry import partial
from returns.pipeline import flow, pipe
from rich.console import Console
from run_logger import RunLogger
from supervised import models
from supervised.lib import create_learning_rate_scheduler, eval_step, train_step
from tensorflow.data import Dataset  # type: ignore
from tensorflow.python.ops.numpy_ops import np_config


def train(
    batch_size: int,
    steps_per_prompt: int,
    data_dir: str,
    disable_jit: bool,
    download_dir: str,
    dropout_rate: float,
    gamma: float,
    learning_rate: float,
    logger: RunLogger,
    max_dataset_step: int,
    num_actions: int,
    num_train_steps: int,
    eval_frequency: int,
    seed: int,
    test_size: int,
):
    if disable_jit:
        from jax._src.config import config

        config.update("jax_disable_jit", True)

    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], "GPU")
    np_config.enable_numpy_behavior()
    console = Console()

    console.log("Loading data...")
    builder_kwargs = dict(
        context_size=steps_per_prompt,
        gamma=gamma,
        max_checkpoint=max_dataset_step,
        test_size=test_size,
    )
    data_dir = flow(
        data_dir,
        Path,
        lambda d: d / "_".join([f"{k}{v}" for k, v in builder_kwargs.items()]),
        str,
    )
    kwargs = dict(name="my_dataset", data_dir=str(data_dir))
    download_and_prepare_kwargs = dict(download_dir=download_dir)
    builder = tfds.builder(**kwargs, **builder_kwargs)  # type: ignore
    builder.download_and_prepare(**download_and_prepare_kwargs)
    ds = tfds.load(
        **kwargs,
        builder_kwargs=builder_kwargs,
        download_and_prepare_kwargs=download_and_prepare_kwargs,
    )
    if batch_size % jax.device_count() > 0:
        raise ValueError("Batch size must be divisible by the number of devices")

    console.log("Creating model...")
    # create the training and development dataset
    config = models.TransformerConfig()
    train_iter = ds["train"]  # type: ignore
    train_iter = tfds.as_numpy(
        train_iter.repeat().batch(batch_size, drop_remainder=True)
    )
    init_batch = flow(train_iter, iter, next)
    model = models.Transformer(
        config=config, dropout_rate=dropout_rate, num_actions=num_actions
    )

    rng = random.PRNGKey(seed)
    rng, init_rng = random.split(rng)

    # call a jitted initialization function to get the initial parameter tree
    @jax.jit
    def initialize_variables(init_rng, init_batch):
        init_variables = model.init(init_rng, inputs=init_batch, train=False)
        return init_variables

    console.log("Initializing variables...")
    init_variables = initialize_variables(init_rng, init_batch)

    learning_rate_fn = create_learning_rate_scheduler(base_learning_rate=learning_rate)

    optimizer = optax.adamw(
        learning_rate_fn, b1=0.9, b2=0.98, eps=1e-9, weight_decay=1e-1
    )
    console.log("Creating train state...")
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_variables["params"], tx=optimizer
    )

    # Replicate optimizer.
    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        partial(train_step, model=model, learning_rate_fn=learning_rate_fn),
        in_axes=0,
        donate_argnums=(0,),
    )

    p_eval_step = jax.pmap(partial(eval_step, model=model), in_axes=0)

    process_metrics = pipe(jnp.mean, lambda x: x.item())

    # We init the first set of dropout PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, jax.local_device_count())
    train_metrics = []
    tick = time.time()
    best_dev_score = 0
    console.log("Training...")
    for step, batch in zip(range(num_train_steps), train_iter):  # type: ignore
        batch = common_utils.shard(batch)
        state, metrics = p_train_step(state, batch, dropout_rng=dropout_rngs)
        train_metrics.append(metrics)
        if (step + 1) % eval_frequency == 0:
            if jax.process_index() == 0:
                steps_per_sec = eval_frequency / (time.time() - tick)
                train_metrics = common_utils.get_metrics(train_metrics)
                train_summary = jax.tree_util.tree_map(process_metrics, train_metrics)
                log = {
                    "run ID": logger.run_id,
                    "steps per second": steps_per_sec,
                    "step": step,
                    "hours": (time.time() - tick) / 3600,
                    **train_summary,
                }
                console.log(log)
                if logger.run_id is not None:
                    logger.log(**log)

            train_metrics = []  # reset metric accumulation for next evaluation cycle.

            eval_metrics = []
            ds_eval = ds["test"]  # type: ignore
            eval_iter = ds_eval.repeat().batch(batch_size, drop_remainder=True)

            for eval_batch in eval_iter:
                eval_batch = jax.tree_util.tree_map(
                    lambda x: x.numpy(), eval_batch
                )  # pylint: disable=protected-access
                if not flow(eval_batch, Dataset.from_tensor_slices, len) == batch_size:
                    breakpoint()
                eval_batch = common_utils.shard(eval_batch)

                metrics = p_eval_step(state.params, eval_batch)

                eval_metrics.append(metrics)
            eval_metrics = common_utils.get_metrics(eval_metrics)
            eval_summary = jax.tree_util.tree_map(process_metrics, eval_metrics)

            if best_dev_score > eval_summary["error"]:
                best_dev_score = eval_summary["error"]
                # TODO: save model.
            eval_summary["best dev score"] = best_dev_score
            if jax.process_index() == 0:
                eval_summary.update(
                    {
                        "run ID": logger.run_id,
                        "step": step,
                        "hours": (time.time() - tick) / 3600,
                    }
                )
                console.log(eval_summary)
                if logger.run_id is not None:
                    logger.log(**eval_summary)
