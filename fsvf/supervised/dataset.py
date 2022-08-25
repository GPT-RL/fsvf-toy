from dataclasses import asdict, dataclass, replace
from functools import reduce
from typing import Any, Iterator

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from etils.epath.abstract_path import Path
from ppo.agent import ExpTuple
from returns.curry import partial
from returns.pipeline import flow
from rich.console import Console
from tensorflow.data import Dataset
from tensorflow_datasets.core import DatasetInfo, GeneratorBasedBuilder, Version
from tensorflow_datasets.core.download import DownloadManager
from tensorflow_datasets.core.features import FeaturesDict, Tensor

console = Console()


@dataclass
class DataPoint:
    time_step: Any
    state: Any
    action: Any
    value: Any

    @staticmethod
    def from_kwargs(time_step, state, action, value, **_):
        return DataPoint(time_step, state, action, value)

    @classmethod
    def from_exp_tuple(cls, time_step, exp_tuple: ExpTuple):
        return cls.from_kwargs(time_step=time_step, **asdict(exp_tuple))


class MyDataset(GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(
        self,
        *args,
        context_size: int,
        gamma: float,
        max_checkpoint: int,
        test_size: int,
        **kwargs,
    ):
        self.context_size = context_size
        self.gamma = gamma
        self.max_checkpoint = max_checkpoint
        self.rng = np.random.default_rng(seed=0)
        self.test_size = test_size
        super().__init__(*args, **kwargs)

    def _info(self) -> DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""

        b = self.context_size + 1
        return DatasetInfo(
            builder=self,
            features=FeaturesDict(
                asdict(
                    DataPoint(
                        time_step=Tensor(shape=[b], dtype=tf.int64),
                        state=Tensor(shape=[b, 5, 5, 3], dtype=tf.int64),
                        action=Tensor(shape=[b], dtype=tf.int64),
                        value=Tensor(shape=[b], dtype=tf.float64),
                    )
                )
            ),
        )

    def _split_generators(self, dl_manager: DownloadManager):
        """Download the data and define splits."""

        def get_checkpoint_idxs():
            for path in dl_manager.download_dir.glob("*/*.npz"):
                checkpoint_idx = int(path.parent.stem)
                if checkpoint_idx <= self.max_checkpoint:
                    yield checkpoint_idx

        checkpoint_idxs = set(get_checkpoint_idxs())
        assert checkpoint_idxs
        assert len(checkpoint_idxs) > self.test_size
        test_idxs = flow(
            checkpoint_idxs,
            list,
            partial(self.rng.choice, size=self.test_size, replace=False),
            set,
        )
        train_idxs = checkpoint_idxs - test_idxs

        def generate(idxs):
            for idx in idxs:
                yield from self._generate_examples(
                    path=dl_manager.download_dir / str(idx)
                )

        return dict(train=generate(train_idxs), test=generate(test_idxs))

    def _generate_examples(self, path: Path) -> Iterator[tuple[str, dict[str, str]]]:
        """Generator of examples for each split."""

        # TODO: implement TFRecordDataset

        def generate_episode():
            # TODO: optimize this loop
            for npzpath in path.glob("*.npz"):
                console.log(npzpath)
                episode = flow(
                    npzpath,
                    np.load,
                    dict,
                    lambda ep: ExpTuple(**ep),
                    lambda exp: replace(exp, state=exp.state.squeeze(1)),
                )
                n = episode.reward.size
                yield flow(
                    np.mgrid[:n, :n],
                    # array([[[0, 0],
                    #         [1, 1]],
                    #        [[0, 1],
                    #         [0, 1]]])
                    partial(np.sum, axis=0),
                    # array([[0, 1],
                    #        [1, 2]])
                    partial(np.flip, axis=0),
                    # array([[1, 2],
                    #        [0, 1]])
                    lambda matrix: matrix - n + 1,
                    # array([[ 0, 1],
                    #        [-1, 0]])
                    partial(np.power, self.gamma),
                    # array([[1.        , 0.9       ],
                    #        [1.11111111, 1.        ]])
                    np.triu,
                    # array([[1., 0.9],
                    #        [0., 1. ]])
                    lambda discounts: discounts @ episode.reward,
                    # episode.reward == array([1, 2])
                    # array([2.8, 2.])
                    lambda value: replace(episode, value=value),
                    partial(DataPoint.from_exp_tuple, np.arange(n)),
                    asdict,
                    Dataset.from_tensor_slices,
                )

        ds = reduce(lambda acc, new: acc.concatenate(new), generate_episode())
        ds = ds.shuffle(len(ds)).batch(1 + self.context_size, drop_remainder=True)
        ds = tfds.as_numpy(ds)
        for ts in ds:  # type: ignore
            dp = DataPoint(**ts)
            key = f"{path.stem}_{str.join('_', dp.time_step.astype(str))}"
            yield key, ts
