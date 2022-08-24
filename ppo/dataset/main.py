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
import os
from pprint import pprint

import dataset  # flake8: noqa
import tensorflow_datasets as tfds
from dollar_lambda import command


@command()
def run(max_checkpoint: int = 50, test_size: int = 100):
    ds_name = "my_dataset"
    tfds.builder(
        ds_name, max_checkpoint=max_checkpoint, test_size=test_size
    ).download_and_prepare(
        download_config=tfds.download.DownloadConfig(),
        download_dir=os.getenv("EXPERIENCE_DIR"),
    )
    df = tfds.load(ds_name)

    for example in df["train"].take(1):
        print("\n\nFirst data point;")
        pprint(example)
        print("\n\n")


if __name__ == "__main__":
    run()
