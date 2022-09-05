import pickle
from dataclasses import asdict, dataclass
from functools import reduce
from typing import Any, Callable, Iterator

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from etils.epath.abstract_path import Path
from ppo.env_utils import MyEnv
from returns.curry import partial
from returns.pipeline import flow
from tensorflow.data import Dataset  # type: ignore
from tensorflow_datasets.core import DatasetInfo, GeneratorBasedBuilder, Version
from tensorflow_datasets.core.features import FeaturesDict, Tensor


@dataclass
class DataPoint:
    state: Any
    action: Any
    value: Any


stack: Callable[[Iterator[Dataset]], Dataset] = partial(
    reduce, lambda acc, new: acc.concatenate(new)
)


class GeneratedDataset(GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(
        self,
        *args,
        context_size: int,
        download_dir: str,
        gamma: float,
        horizon: int,
        num_generated_examples: int,
        **kwargs,
    ):
        self.context_size = context_size
        self.gamma = gamma
        self.horizon = horizon
        self.num_generated_examples = num_generated_examples
        self.rng = np.random.default_rng(seed=0)
        with Path(download_dir, "observation.pkl").open("rb") as f:
            self.observation_space = pickle.load(f)
        self.num_actions = len(MyEnv.deltas)

        super().__init__(*args, **kwargs)

    def _info(self) -> DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        b = self.context_size + 1
        return DatasetInfo(
            builder=self,
            disable_shuffling=True,
            features=FeaturesDict(
                asdict(
                    DataPoint(
                        state=Tensor(
                            shape=[self.num_actions, b, *self.observation_space.shape],
                            dtype=tf.int64,
                        ),
                        action=Tensor(shape=[self.num_actions, b], dtype=tf.int64),
                        value=Tensor(shape=[self.num_actions, b], dtype=tf.float64),
                    )
                )
            ),
        )

    def _generate_examples(self, _):
        def make_ds(poss: np.ndarray, goals: np.ndarray, actions: np.ndarray):
            *state_shape, _ = self.observation_space.shape

            new_poss = poss + MyEnv.deltas[actions]
            new_poss = np.clip(new_poss, [0, 0], np.array(state_shape) - 1)
            distances = flow(new_poss - goals, np.abs, partial(np.sum, axis=1))
            values = self.gamma ** (1 + distances) * (distances < self.horizon)
            states = np.zeros(
                (len(poss), *self.observation_space.shape), dtype=np.int64
            )
            for state in states:
                assert self.observation_space.contains(state)
            arange = np.arange(len(poss))
            states[arange, poss[:, 0], poss[:, 1], 1] = 1
            states[arange, goals[:, 0], goals[:, 1], 2] = 1
            return flow(
                DataPoint(states, actions, values),
                asdict,
                Dataset.from_tensor_slices,
            )

        for _ in range(self.num_generated_examples):
            *state_shape, _ = self.observation_space.shape
            poss = self.rng.integers(
                low=0, high=state_shape, size=[self.context_size + 1, 2]
            )
            goals = self.rng.integers(
                low=0, high=state_shape, size=[self.context_size + 1, 2]
            )
            actions = self.rng.integers(
                low=0, high=self.num_actions, size=[self.context_size]
            )
            context = make_ds(poss[:-1], goals[:-1], actions)

            def add_query(action: int):
                query = make_ds(poss[-1:], goals[-1:], np.array([action]))
                dp = context.concatenate(query)
                return dp.batch(len(dp))

            for dp in flow(
                self.num_actions,
                range,
                partial(map, add_query),
                stack,
                lambda d: d.batch(self.num_actions),
                tfds.as_numpy,
            ):
                yield str(self.rng.bit_generator.state["state"]), dp

    def _split_generators(self, _):
        """Download the data and define splits."""
        return {"test": self._generate_examples(None)}
