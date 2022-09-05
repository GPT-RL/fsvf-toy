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
class DataPoint:
    X: Any
    Y: Any


class LinearTransformationDataset(GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(
        self,
        *args,
        n_context: int,
        n_dim: int,
        n_test: int,
        n_train: int,
        **kwargs,
    ):
        self.n_context = n_context
        self.n_dim = n_dim
        self.n_test = n_test
        self.n_train = n_train
        self.rng = np.random.default_rng(seed=0)
        super().__init__(*args, **kwargs)

    def _info(self) -> DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""

        return DatasetInfo(
            builder=self,
            features=FeaturesDict(
                asdict(
                    DataPoint(
                        X=Tensor(shape=[self.n_context, self.n_dim], dtype=tf.float64),
                        Y=Tensor(shape=[self.n_context], dtype=tf.float64),
                    )
                )
            ),
        )

    def _generate_examples(self, W: np.ndarray) -> Iterator[tuple[str, dict[str, str]]]:
        """Generator of examples for each split."""

        for w in W:
            X = self.rng.random(size=w.shape)
            Y = np.sum(X * w, axis=-1)
            yield f"{X}{Y}", asdict(DataPoint(X=X, Y=Y))

    def _split_generators(self, dl_manager: DownloadManager):
        """Download the data and define splits."""
        W = self.rng.random((self.n_train + self.n_test, self.n_context, self.n_dim))
        test, train = np.split(W, [self.n_test], axis=0)

        return dict(
            train=self._generate_examples(train),
            test=self._generate_examples(test),
        )
