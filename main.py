#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging
import re
import tensorflow_datasets as tfds
from dataset import myrlu

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


def run(argv=None, save_main_session=True):
    """Main entry point; defines and runs the wordcount pipeline."""

    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    tfds.builder("my_mnist").download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            beam_options=PipelineOptions(
                [
                    "--runner=SparkRunner",
                    "--spark_version=3",
                ]
            )
        )
    )
    df = tfds.load("my_mnist")

    for example in df["train"].take(1):
        image = example["image"]
        label = example["label"]
        print("image:", image.shape)
        print("label:", label)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
