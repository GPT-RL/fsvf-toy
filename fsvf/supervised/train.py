import logging
import time
from contextlib import contextmanager
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import supervised.ppo_dataset  # noqa: F401
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import jax_utils
from flax.training import common_utils, train_state
from jax import random
from returns.curry import partial
from returns.pipeline import flow, pipe
from rich.console import Console
from rich.logging import RichHandler
from run_logger import RunLogger
from supervised import models
from supervised.input_pipeline import get_generated_dataset, get_ppo_dataset
from supervised.lib import (
    create_learning_rate_scheduler,
    test_generated_step,
    test_ppo_step,
    train_step,
)
from tensorflow.data import Dataset  # type: ignore
from tensorflow.python.ops.numpy_ops import np_config


def train(
    batch_size: int,
    data_dir: str,
    disable_jit: bool,
    download_dir: str,
    dropout_rate: float,
    gamma: float,
    learning_rate: float,
    log_level: str,
    max_dataset_step: int,
    num_actions: int,
    num_generated_examples: int,
    num_train_steps: int,
    run_logger: RunLogger,
    seed: int,
    steps_per_prompt: int,
    test_frequency: int,
    test_size: int,
):
    if disable_jit:
        from jax._src.config import config

        config.update("jax_disable_jit", True)

    def pmap(
        fun,
        *args,
        donate_argnums=(),
        **kwargs,
    ):
        return (
            jax.vmap(fun, *args, **kwargs)
            if disable_jit
            else jax.pmap(fun, *args, donate_argnums=donate_argnums, **kwargs)
        )

    logging.basicConfig(datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], "GPU")
    np_config.enable_numpy_behavior()

    if batch_size % jax.device_count() > 0:
        raise ValueError("Batch size must be divisible by the number of devices")

    console = Console()

    @contextmanager
    def timer(msg: str, stacklevel=3):
        logger.info(msg, stacklevel=stacklevel)
        tick = time.time()
        yield
        logger.info(f"Took {time.time() - tick:.2f} seconds.", stacklevel=stacklevel)

    def preprocess_data(ds: Dataset, repeat: Optional[int]):
        return tfds.as_numpy(
            ds.shuffle(len(ds))  # TODO: try removing this
            .cache()  # cache the dataset in memory and repeat.
            .repeat(repeat)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    with timer("Loading data..."):
        ppo_datasets = get_ppo_dataset(
            data_dir=data_dir,
            download_dir=download_dir,
            gamma=gamma,
            max_dataset_step=max_dataset_step,
            test_size=test_size,
            steps_per_prompt=steps_per_prompt,
        )
        train_iter = preprocess_data(ppo_datasets["train"], repeat=None)
        generated_datasets = get_generated_dataset(
            data_dir=data_dir,
            download_dir=download_dir,
            gamma=gamma,
            num_generated_examples=num_generated_examples,
            steps_per_prompt=steps_per_prompt,
        )
        generated_dataset = generated_datasets["test"]  # stupid tfds
        generated_iter = tfds.as_numpy(
            generated_dataset.cache()  # cache the dataset in memory and repeat.
            .repeat(1)
            .batch(jax.device_count(), drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    init_batch = flow(train_iter, iter, next)

    config = models.TransformerConfig()
    model = models.Transformer(
        config=config, dropout_rate=dropout_rate, num_actions=num_actions
    )

    rng = random.PRNGKey(seed)
    rng, init_rng = random.split(rng)

    # call a jitted initialization function to get the initial parameter tree
    @jax.jit
    def initialize_variables(init_rng, init_batch):
        return model.init(init_rng, inputs=init_batch, train=False)

    with timer("Initializing variables..."):
        init_variables = initialize_variables(init_rng, init_batch)

    learning_rate_fn = create_learning_rate_scheduler(base_learning_rate=learning_rate)

    optimizer = optax.adamw(
        learning_rate_fn, b1=0.9, b2=0.98, eps=1e-9, weight_decay=1e-1
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_variables["params"], tx=optimizer
    )

    # Replicate optimizer.
    state = jax_utils.replicate(state)

    p_train_step = pmap(
        partial(train_step, model=model, learning_rate_fn=learning_rate_fn),
        axis_name="batch",
        donate_argnums=(0,),
    )

    p_test_ppo_step = pmap(partial(test_ppo_step, model=model), axis_name="batch")
    p_test_generated_step = pmap(
        partial(test_generated_step, model=model), axis_name="batch"
    )

    process_metrics = pipe(
        common_utils.get_metrics,
        partial(jax.tree_util.tree_map, pipe(jnp.mean, lambda x: x.item())),
    )

    # We init the first set of dropout PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, jax.local_device_count())
    best_dev_score = 0
    init_batch = common_utils.shard(init_batch)
    with timer("Jitting train step..."):
        state, _ = p_train_step(state, init_batch, dropout_rng=dropout_rngs)
    logger.info("Training...")
    train_metrics = []
    tick = time.time()
    for step, batch in zip(range(num_train_steps), train_iter):  # type: ignore
        batch = common_utils.shard(batch)
        state, metrics = p_train_step(state, batch, dropout_rng=dropout_rngs)
        train_metrics.append(metrics)
        if step % test_frequency == 0:
            ppo_test_iter = preprocess_data(ppo_datasets["test"], repeat=1)

            with timer("Evaluating..."):
                accuracies = []
                for batch in generated_iter:  # type: ignore
                    comparisons = pipe(
                        lambda v: v[:, :, -1],
                        lambda v: np.expand_dims(v, -1) < np.expand_dims(v, -2),
                    )
                    estimate = flow(
                        p_test_generated_step(state.params, batch),
                        jax.device_get,
                        comparisons,
                    )
                    target = comparisons(batch["value"])
                    accuracy = np.mean(estimate == target)
                    accuracies.append(accuracy)
                mean_accuracy = np.mean(accuracies)

                test_ppo_metrics = []
                for batch in ppo_test_iter:  # type: ignore
                    assert flow(batch, Dataset.from_tensor_slices, len) == batch_size
                    batch = common_utils.shard(batch)
                    metrics = p_test_ppo_step(state.params, batch)
                    test_ppo_metrics.append(
                        {f"test {k}": v for k, v in metrics.items()}
                    )
                test_ppo_summary = process_metrics(test_ppo_metrics)
            train_summary = process_metrics(train_metrics)
            train_metrics = []
            if best_dev_score > test_ppo_summary["test error"]:
                best_dev_score = test_ppo_summary["test error"]
                # TODO: save model.
            test_ppo_summary["best dev score"] = best_dev_score
            if jax.process_index() == 0:
                steps_per_sec = step / (time.time() - tick)
                log = {
                    "accuracy": mean_accuracy,
                    "run ID": run_logger.run_id,
                    "hours": (time.time() - tick) / 3600,
                    "step": step,
                    "steps per second": steps_per_sec,
                    **test_ppo_summary,
                    **train_summary,
                }
                console.log(log)
                if run_logger.run_id is not None:
                    run_logger.log(**log)
