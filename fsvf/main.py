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
from pprint import pprint

import tensorflow_datasets as tfds
from apache_beam.options.pipeline_options import PipelineOptions
from dataset import my_rlu


def run(argv=None, save_main_session=True):
    tfds.builder("my_rlu").download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            beam_options=PipelineOptions(
                [
                    "--runner=SparkRunner",
                    "--spark_version=3",
                ]
            )
        ),
    )
    df = tfds.load("my_rlu")

    for example in df["train"].take(1):
        print("\n\nFirst data point;")
        pprint(example)
        print("\n\n")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
