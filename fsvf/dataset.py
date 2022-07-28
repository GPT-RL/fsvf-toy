# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
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

"""RLU Atari datasets."""

import math
import os
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.rl_unplugged import atari_utils

_DESCRIPTION = """
RL Unplugged is suite of benchmarks for offline reinforcement learning. The RL
Unplugged is designed around the following considerations: to facilitate ease of
use, we provide the datasets with a unified API which makes it easy for the
practitioner to work with all data in the suite once a general pipeline has been
established.

The datasets follow the [RLDS format](https://github.com/google-research/rlds)
to represent steps and episodes.

"""


_HOMEPAGE = "https://github.com/deepmind/deepmind-research/tree/master/rl_unplugged"


# Note that rewards and episode_return are actually also clipped.
_FEATURE_DESCRIPTION = {
    "checkpoint_idx": tf.io.FixedLenFeature([], tf.int64),
    "episode_idx": tf.io.FixedLenFeature([], tf.int64),
    "episode_return": tf.io.FixedLenFeature([], tf.float32),
    "clipped_episode_return": tf.io.FixedLenFeature([], tf.float32),
    "observations": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    "actions": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "unclipped_rewards": tf.io.FixedLenSequenceFeature(
        [], tf.float32, allow_missing=True
    ),
    "clipped_rewards": tf.io.FixedLenSequenceFeature(
        [], tf.float32, allow_missing=True
    ),
    "discounts": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
}

EPISODES_PER = 8
DISCOUNT = 0.99


def filename(prefix: str, num_shards: int, shard_id: int):
    return os.fspath(tfds.core.Path(f"{prefix}-{shard_id:05d}-of-{num_shards:05d}"))


def get_files(prefix: str, num_shards: int) -> List[str]:
    return [
        filename(prefix, num_shards, i) for i in range(num_shards)
    ]  # pytype: disable=bad-return-type  # gen-stub-imports


def float_tensor_feature(size: int) -> tfds.features.Tensor:
    return tfds.features.Tensor(
        shape=(size,), dtype=tf.float32, encoding=tfds.features.Encoding.ZLIB
    )


class MyRLU(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for RLU Atari."""

    _SHARDS = 50
    _INPUT_FILE_PREFIX = "gs://rl_unplugged/atari_episodes_ordered"

    VERSION = tfds.core.Version("1.3.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "Added is_last.",
        "1.2.0": "Added checkpoint id.",
        "1.3.0": "Removed redundant clipped reward fields.",
    }

    BUILDER_CONFIGS = atari_utils.builder_configs()

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION + self.get_description(),
            features=self.get_features_dict(),
            supervised_keys=None,  # disabled
            homepage=_HOMEPAGE,
            citation=self.get_citation(),
        )

    def get_file_prefix(self):
        run = self.builder_config.run
        game = self.builder_config.game
        return atari_utils.file_prefix(self._INPUT_FILE_PREFIX, run, game)

    def get_citation(self):
        return atari_utils.citation()

    def get_description(self):
        return atari_utils.description()

    def num_shards(self):
        return atari_utils.num_shards(self.builder_config.game, self._SHARDS)

    def get_features_dict(self):
        return tfds.features.FeaturesDict(
            dict(
                checkpoint_id=tf.int64,
                episodes=tfds.features.Dataset(
                    dict(
                        steps=tfds.features.Dataset(
                            dict(
                                observation=tfds.features.Image(
                                    shape=(
                                        84,
                                        84,
                                        1,
                                    ),
                                    dtype=tf.uint8,
                                    encoding_format="png",
                                ),
                                action=tf.int64,
                                return_to_go=tfds.features.Scalar(
                                    dtype=tf.float32,
                                    doc=tfds.features.Documentation(
                                        desc="Discounted sum of future rewards."
                                    ),
                                ),
                                reward=tfds.features.Scalar(
                                    dtype=tf.float32,
                                    doc=tfds.features.Documentation(
                                        desc="Clipped reward.", value_range="[-1, 1]"
                                    ),
                                ),
                                is_terminal=tf.bool,
                                is_first=tf.bool,
                                is_last=tf.bool,
                            )
                        ),
                        episode_id=tf.int64,
                        episode_return=tfds.features.Scalar(
                            dtype=tf.float32,
                            doc=tfds.features.Documentation(
                                desc="Sum of the clipped rewards."
                            ),
                        ),
                    )
                ),
            )
        )

    def get_episode_id(self, episode):
        return atari_utils.episode_id(episode)

    def tf_example_to_step_ds(
        self, tf_example: tf.train.Example
    ) -> Tuple[Dict[str, Any], int]:
        """Generates an RLDS episode from an Atari TF Example.
        Args:
          tf_example: example from an Atari dataset.
        Returns:
          RLDS episode.
        """

        data = tf.io.parse_single_example(tf_example, _FEATURE_DESCRIPTION)
        episode_length = tf.size(data["actions"])
        is_first = tf.concat([[True], [False] * tf.ones(episode_length - 1)], axis=0)
        is_last = tf.concat([[False] * tf.ones(episode_length - 1), [True]], axis=0)

        is_terminal = [False] * tf.ones_like(data["actions"])
        discounts = data["discounts"]
        if discounts[-1] == 0.0:
            is_terminal = tf.concat(
                [[False] * tf.ones(episode_length - 1, tf.int64), [True]], axis=0
            )
            # If the episode ends in a terminal state, in the last step only the
            # observation has valid information (the terminal state).
            discounts = tf.concat([discounts[1:], [0.0]], axis=0)
        episode = {
            # Episode Metadata
            "episode_id": data["episode_idx"],
            # "checkpoint_id": data["checkpoint_idx"],
            "episode_return": data["episode_return"],
            "steps": {
                "observation": data["observations"],
                "action": data["actions"],
                "reward": data["unclipped_rewards"],
                # "discount": discounts,
                "is_first": is_first,
                "is_last": is_last,
                "is_terminal": is_terminal,
            },
        }
        return episode, data["checkpoint_idx"]

    def get_splits(self):
        paths = {
            "file_paths": get_files(
                prefix=self.get_file_prefix(), num_shards=self.num_shards()
            ),
        }
        return {
            "train": self._generate_examples(paths),
        }

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        del dl_manager

        return self.get_splits()

    def generate_examples_one_file(
        self, path
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Yields examples from one file."""
        # Dataset of tf.Examples containing full episodes.
        example_ds = tf.data.TFRecordDataset(
            filenames=str(path), compression_type="GZIP"
        )
        # Dataset of episodes, each represented as a dataset of steps.
        episode_ds = example_ds.map(
            self.tf_example_to_step_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        episode_ds = iter(tfds.as_numpy(episode_ds))
        while True:
            try:
                chunk = [next(episode_ds) for _ in range(EPISODES_PER)]
            except StopIteration:
                return

            episodes, file_checkpoint_ids = zip(*chunk)
            [checkpoint_id] = set(file_checkpoint_ids)
            episode_id = min(ep["episode_id"] for ep in episodes)
            record_id = f"{checkpoint_id}_{episode_id}"
            for ep in episodes:
                rewards = ep["steps"]["reward"]
                n1 = math.ceil(len(rewards) / 2) - 1
                n2 = len(rewards) // 2
                powers = np.mgrid[-n1 : len(rewards) - n1, -n2 : len(rewards) - n2]
                powers = powers.sum(0)
                powers = np.flip(powers, axis=0)
                """
                powers:
                [  0  1  2 ... ]
                [ -1  0  1 ... ]
                [ -2 -1  0 ... ]
                ...
                """
                discounts = DISCOUNT**powers
                discounts = discounts * (powers >= 0)
                """
                dicsounts:
                [  1.00  0.99  0.98 ... ]
                [  0.00  1.00  0.99 ... ]
                [  0.00  0.00  1.00 ... ]
                ...
                """
                assert discounts[-1, -1] == 1
                assert np.all(discounts[-1, :-1] == 0)
                rewards = np.expand_dims(rewards, 0)
                return_to_go = np.sum(rewards * discounts, axis=1)
                ep["steps"]["return_to_go"] = return_to_go
            episodes = []
            yield record_id, dict(checkpoint_id=checkpoint_id, episodes=episodes)

    def _generate_examples(self, paths):
        """Yields examples."""
        beam = tfds.core.lazy_imports.apache_beam
        file_paths = paths["file_paths"]
        # for p in file_paths:
        # assert "Alien" in inp
        file_paths = file_paths[:1]  # TODO
        return beam.Create(file_paths) | beam.FlatMap(self.generate_examples_one_file)
