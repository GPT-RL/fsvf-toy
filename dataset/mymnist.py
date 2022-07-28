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

"""MNIST, Fashion MNIST, KMNIST and EMNIST."""

import os
import numpy as np
from six.moves import urllib
import tensorflow as tf

import tensorflow_datasets.public_api as tfds

# MNIST constants
# CVDF mirror of http://yann.lecun.com/exdb/mnist/
_MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_MNIST_TRAIN_DATA_FILENAME = "train-images-idx3-ubyte.gz"
_MNIST_TRAIN_LABELS_FILENAME = "train-labels-idx1-ubyte.gz"
_MNIST_TEST_DATA_FILENAME = "t10k-images-idx3-ubyte.gz"
_MNIST_TEST_LABELS_FILENAME = "t10k-labels-idx1-ubyte.gz"
_MNIST_IMAGE_SIZE = 28
MNIST_IMAGE_SHAPE = (_MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1)
MNIST_NUM_CLASSES = 10
_TRAIN_EXAMPLES = 60000
_TEST_EXAMPLES = 10000

_MNIST_CITATION = """\
@article{lecun2010mnist,
  title={MNIST handwritten digit database},
  author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
  journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
  volume={2},
  year={2010}
}
"""


class MyMNIST(tfds.core.GeneratorBasedBuilder):
    """MyMNIST."""

    URL = _MNIST_URL

    VERSION = tfds.core.Version("3.0.1")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("The MNIST database of handwritten digits."),
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=MNIST_IMAGE_SHAPE),
                    "label": tfds.features.ClassLabel(num_classes=MNIST_NUM_CLASSES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="http://yann.lecun.com/exdb/mnist/",
            citation=_MNIST_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Download the full MNIST Database
        filenames = {
            "train_data": _MNIST_TRAIN_DATA_FILENAME,
            "train_labels": _MNIST_TRAIN_LABELS_FILENAME,
            "test_data": _MNIST_TEST_DATA_FILENAME,
            "test_labels": _MNIST_TEST_LABELS_FILENAME,
        }
        mnist_files = dl_manager.download_and_extract(
            {k: urllib.parse.urljoin(self.URL, v) for k, v in filenames.items()}
        )

        # MNIST provides TRAIN and TEST splits, not a VALIDATION split, so we only
        # write the TRAIN and TEST splits to disk.
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(
                    num_examples=_TRAIN_EXAMPLES,
                    data_path=mnist_files["train_data"],
                    label_path=mnist_files["train_labels"],
                ),
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs=dict(
                    num_examples=_TEST_EXAMPLES,
                    data_path=mnist_files["test_data"],
                    label_path=mnist_files["test_labels"],
                ),
            ),
        ]

    def _generate_examples(self, num_examples, data_path, label_path):
        """Generate MNIST examples as dicts.
        Args:
          num_examples (int): The number of example.
          data_path (str): Path to the data files
          label_path (str): Path to the labels
        Yields:
          Generator yielding the next examples
        """

        images = _extract_mnist_images(data_path, num_examples)
        labels = _extract_mnist_labels(label_path, num_examples)
        data = list(zip(images, labels))
        # Using index as key since data is always loaded in same order.

        beam = tfds.core.lazy_imports.apache_beam

        def _process_example(index_image_label):
            index, (image, label) = index_image_label
            # Use filename as key
            return index, {"image": image, "label": label}

        return beam.Create(list(enumerate(data))) | beam.Map(_process_example)


def _extract_mnist_images(image_filepath, num_images):
    with tf.io.gfile.GFile(image_filepath, "rb") as f:
        f.read(16)  # header
        buf = f.read(_MNIST_IMAGE_SIZE * _MNIST_IMAGE_SIZE * num_images)
        data = np.frombuffer(
            buf,
            dtype=np.uint8,
        ).reshape(num_images, _MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1)
        return data


def _extract_mnist_labels(labels_filepath, num_labels):
    with tf.io.gfile.GFile(labels_filepath, "rb") as f:
        f.read(8)  # header
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels
