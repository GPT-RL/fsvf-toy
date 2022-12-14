import pickle
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterator

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from etils.epath.abstract_path import Path
from ppo.agent import ExpTuple
from returns.curry import partial
from returns.pipeline import flow
from rich.console import Console
from supervised import generated_dataset
from tensorflow.data import Dataset  # type: ignore
from tensorflow_datasets.core import DatasetInfo, GeneratorBasedBuilder, Version
from tensorflow_datasets.core.download import DownloadManager
from tensorflow_datasets.core.features import FeaturesDict, Tensor

console = Console()


@dataclass
class DataPoint(generated_dataset.DataPoint):
    action: Any
    time_step: Any

    @staticmethod
    def from_kwargs(*_, state, action, value, time_step, **__):
        return DataPoint(state=state, action=action, value=value, time_step=time_step)

    @classmethod
    def from_exp_tuple(cls, time_step, exp_tuple: ExpTuple):
        return cls.from_kwargs(time_step=time_step, **asdict(exp_tuple))


class PpoDataset(GeneratorBasedBuilder):
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
        max_checkpoint: int,
        test_size: int,
        **kwargs,
    ):
        self.context_size = context_size
        self.gamma = gamma
        self.horizon = horizon
        self.max_checkpoint = max_checkpoint
        self.rng = np.random.default_rng(seed=0)
        self.test_size = test_size
        with Path(download_dir, "observation.pkl").open("rb") as f:
            self.observation_space = pickle.load(f)
        super().__init__(*args, **kwargs)

    def _info(self) -> DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""

        b = self.context_size + 1
        return DatasetInfo(
            builder=self,
            features=FeaturesDict(
                asdict(
                    generated_dataset.DataPoint(
                        state=Tensor(
                            shape=[b, *self.observation_space.shape], dtype=tf.int64
                        ),
                        value=Tensor(shape=[b], dtype=tf.float64),
                    )
                )
            ),
        )

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
                    partial(np.tril, k=self.horizon),
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

        def key_f(row):
            return tf.cast(row["action"], tf.int64)

        ds = generated_dataset.stack(generate_episode())
        length = len(ds)
        ds = ds.shuffle(len(ds)).group_by_window(  # shuffle within batch
            key_f,
            lambda _, n: n.batch(1 + self.context_size, drop_remainder=True),
            window_size=length,
        )
        ds = ds.shuffle(length)  # shuffle between batches

        for ts in tfds.as_numpy(ds):  # type: ignore
            dp = DataPoint(**ts)
            del ts["action"]
            del ts["time_step"]
            assert np.unique(dp.action).size == 1
            key = f"{path.stem}_{str.join('_', dp.time_step.astype(str))}"
            yield key, ts

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

        def generate_from_data(idxs):
            for idx in idxs:
                yield from self._generate_examples(
                    path=dl_manager.download_dir / str(idx)
                )

        return dict(
            train=generate_from_data(train_idxs),
            test=generate_from_data(test_idxs),
        )
