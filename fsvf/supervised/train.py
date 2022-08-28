import logging
import time
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import optax
import supervised.dataset  # noqa: F401
import tensorflow as tf
from flax import jax_utils
from flax.training import common_utils, train_state
from jax import random
from returns.curry import partial
from returns.pipeline import flow, pipe
from rich.console import Console
from rich.logging import RichHandler
from run_logger import RunLogger
from supervised import models
from supervised.input_pipeline import trajectory_dataset
from supervised.lib import create_learning_rate_scheduler, eval_step, train_step
from tensorflow.data import Dataset  # type: ignore
from tensorflow.python.ops.numpy_ops import np_config


def train(
    batch_size: int,
    data_dir: str,
    disable_jit: bool,
    dev: str,
    download_dir: str,
    dropout_rate: float,
    gamma: float,
    learning_rate: float,
    log_level: str,
    max_dataset_step: int,
    num_actions: int,
    num_train_steps: int,
    run_logger: RunLogger,
    eval_frequency: int,
    seed: int,
    steps_per_prompt: int,
    test_size: int,
    train: str,
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

    with timer("Loading data..."):
        train_iter = trajectory_dataset(
            batch_size=batch_size,
            data_dir=data_dir,
            download_dir=download_dir,
            gamma=gamma,
            max_dataset_step=max_dataset_step,
            test_size=test_size,
            split="train",
            steps_per_prompt=steps_per_prompt,
            repeat=None,
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
        init_variables = initialize_variables(init_rng, init_batch["action"])

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

    p_eval_step = pmap(partial(eval_step, model=model), axis_name="batch")

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
        if (step + 1) % eval_frequency == 0:
            eval_metrics = []
            eval_iter = trajectory_dataset(
                batch_size=batch_size,
                data_dir=data_dir,
                download_dir=download_dir,
                gamma=gamma,
                max_dataset_step=max_dataset_step,
                test_size=test_size,
                split="test",
                steps_per_prompt=steps_per_prompt,
                repeat=1,
            )

            with timer("Evaluating..."):
                for eval_batch in eval_iter:  # type: ignore
                    assert (
                        flow(eval_batch, Dataset.from_tensor_slices, len) == batch_size
                    )
                    eval_batch = common_utils.shard(eval_batch)
                    metrics = p_eval_step(state.params, eval_batch)
                    eval_metrics.append({f"test {k}": v for k, v in metrics.items()})
            eval_summary = process_metrics(eval_metrics)
            train_summary = process_metrics(train_metrics)
            train_metrics = []
            if best_dev_score < eval_summary["test accuracy"]:
                best_dev_score = eval_summary["test accuracy"]
                # TODO: save model.
            eval_summary["best dev score"] = best_dev_score
            if jax.process_index() == 0:
                steps_per_sec = step / (time.time() - tick)
                log = {
                    "run ID": run_logger.run_id,
                    "hours": (time.time() - tick) / 3600,
                    "step": step,
                    "steps per second": steps_per_sec,
                    **eval_summary,
                    **train_summary,
                }
                console.log(log)
                if run_logger.run_id is not None:
                    run_logger.log(**log)
