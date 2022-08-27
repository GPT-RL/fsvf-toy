import logging
import time
from contextlib import contextmanager
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
from rich.logging import RichHandler
from run_logger import RunLogger
from supervised import input_pipeline, models
from supervised.lib import (
    compute_metrics,
    create_learning_rate_scheduler,
    eval_step,
    pad_examples,
    train_step,
)
from supervised.models import TransformerConfig
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
    max_len: int,
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

    logging.basicConfig(datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], "GPU")
    np_config.enable_numpy_behavior()

    if batch_size % jax.device_count() > 0:
        raise ValueError("Batch size must be divisible by the number of devices")
    vocabs = input_pipeline.create_vocabs(train)

    console = Console()

    @contextmanager
    def timer(msg: str, stacklevel=3):
        logger.info(msg, stacklevel=stacklevel)
        tick = time.time()
        yield
        logger.info(f"Took {time.time() - tick:.2f} seconds.", stacklevel=stacklevel)

    with timer("Loading data..."):
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
    # create the training and development dataset
    config = TransformerConfig(
        vocab_size=len(vocabs["forms"]),
        output_vocab_size=len(vocabs["xpos"]),
        max_len=max_len,
    )
    attributes_input = [input_pipeline.CoNLLAttributes.FORM]
    attributes_target = [input_pipeline.CoNLLAttributes.XPOS]
    train_ds = input_pipeline.sentence_dataset_dict(
        train,
        vocabs,
        attributes_input,
        attributes_target,
        batch_size=batch_size,
        bucket_size=config.max_len,
    )
    train_iter = iter(tfds.as_numpy(train_ds))
    eval_ds = input_pipeline.sentence_dataset_dict(
        dev,
        vocabs,
        attributes_input,
        attributes_target,
        batch_size=batch_size,
        bucket_size=config.max_len,
        repeat=1,
    )
    init_batch = flow(train_iter, iter, next, lambda x: x["inputs"])
    model = models.Transformer(config)

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

    p_train_step = jax.pmap(
        partial(train_step, model=model, learning_rate_fn=learning_rate_fn),
        axis_name="batch",
        donate_argnums=(0,),
    )

    def eval_step(params, batch):
        """Calculate evaluation metrics on a batch."""
        inputs, targets = batch["inputs"], batch["targets"]
        weights = jnp.where(targets > 0, 1.0, 0.0)
        logits = model.apply({"params": params}, inputs=inputs, train=False)
        return compute_metrics(logits, targets, weights)

    p_eval_step = jax.pmap(eval_step, axis_name="batch")

    # We init the first set of dropout PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, jax.local_device_count())
    metrics_all = []
    tick = time.time()
    best_dev_score = 0
    for step, batch in zip(range(num_train_steps), train_iter):
        batch = common_utils.shard(batch)

        state, metrics = p_train_step(state, batch, dropout_rng=dropout_rngs)
        metrics_all.append(metrics)
        if (step + 1) % eval_frequency == 0:
            metrics_all = common_utils.get_metrics(metrics_all)
            lr = metrics_all.pop("learning_rate").mean()
            metrics_sums = jax.tree_util.tree_map(jnp.sum, metrics_all)
            denominator = metrics_sums.pop("denominator")
            summary = jax.tree_util.tree_map(
                lambda x: x / denominator, metrics_sums
            )  # pylint: disable=cell-var-from-loop
            summary["learning_rate"] = lr
            if jax.process_index() == 0:
                steps_per_sec = eval_frequency / (time.time() - tick)
                log = {
                    "run ID": run_logger.run_id,
                    "steps per second": steps_per_sec,
                    "step": step,
                }
                console.log(log)
                if run_logger.run_id is not None:
                    run_logger.log(**log)

            metrics_all = []  # reset metric accumulation for next evaluation cycle.

            eval_metrics = []
            eval_iter = iter(eval_ds)
            # eval_iter = iter(ds["test"])

            for eval_batch in eval_iter:
                eval_batch = jax.tree_util.tree_map(
                    lambda x: x._numpy(), eval_batch
                )  # pylint: disable=protected-access
                # Handle final odd-sized batch by padding instead of dropping it.
                cur_pred_batch_size = eval_batch["inputs"].shape[0]
                if cur_pred_batch_size != batch_size:
                    # pad up to batch size
                    eval_batch = jax.tree_util.tree_map(
                        lambda x: pad_examples(x, batch_size), eval_batch
                    )
                eval_batch = common_utils.shard(eval_batch)

                metrics = p_eval_step(state.params, eval_batch)

                eval_metrics.append(metrics)
            eval_metrics = common_utils.get_metrics(eval_metrics)
            eval_metrics_sums = jax.tree_util.tree_map(jnp.sum, eval_metrics)
            eval_denominator = eval_metrics_sums.pop("denominator")
            eval_summary = jax.tree_util.tree_map(
                lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
                eval_metrics_sums,
            )

            if best_dev_score < eval_summary["accuracy"]:
                best_dev_score = eval_summary["accuracy"]
                # TODO: save model.
            eval_summary["best dev score"] = best_dev_score
            if jax.process_index() == 0:
                log = {k: v.to_py().item() for k, v in eval_summary.items()}
                log.update(
                    {
                        "run ID": run_logger.run_id,
                        "step": step,
                        "hours": (time.time() - tick) / 3600,
                    }
                )
                console.log(log)
                if run_logger.run_id is not None:
                    run_logger.log(**log)
