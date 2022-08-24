import itertools
import re
from typing import Iterator

import numpy as np
import tensorflow as tf
from etils.epath.abstract_path import Path
from returns.curry import partial
from returns.pipeline import flow, pipe
from tensorflow_datasets.core import DatasetInfo, GeneratorBasedBuilder, Version
from tensorflow_datasets.core.download import DownloadManager
from tensorflow_datasets.core.features import Dataset, FeaturesDict, Scalar, Tensor


class MyDataset(GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, max_checkpoint: int, test_size: int):
        self.max_checkpoint = max_checkpoint
        self.rng = np.random.default_rng(seed=0)
        self.test_size = test_size
        super().__init__()

    def _info(self) -> DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        state = Tensor(shape=(5, 5, 3), dtype=tf.int32)
        action = Scalar(dtype=tf.int32)
        value = Scalar(dtype=tf.float32)
        context = dict(state=state, action=action, value=value)
        query = dict(state=state, action=action)

        return DatasetInfo(
            builder=self,
            features=FeaturesDict(dict(context=Dataset(context), **query, value=value)),
        )

    def _split_generators(self, dl_manager: DownloadManager):
        """Download the data and define splits."""

        def get_checkpoint_idxs():
            for path in dl_manager.download_dir.glob("*.npz"):
                [checkpoint_idx] = re.findall(r"(\d+).\d+", path.stem)
                checkpoint_idx = int(checkpoint_idx)
                if checkpoint_idx > self.max_checkpoint:
                    yield checkpoint_idx

        checkpoint_idxs = set(get_checkpoint_idxs())
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
                    path=dl_manager.download_dir / f"{idx}.npz"
                )

        return dict(train=generate(train_idxs), test=generate(test_idxs))

    def _generate_examples(self, path: Path) -> Iterator[tuple[str, dict[str, str]]]:
        """Generator of examples for each split."""
        for img_path in path.glob("*.jpeg"):
            # Yields (key, example)
            yield img_path.name, {
                "image": img_path,
                "label": "yes" if img_path.name.startswith("yes_") else "no",
            }
