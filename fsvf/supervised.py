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
from pprint import pprint
from typing import Optional

import supervised.dataset as dataset  # noqa: F401
import tensorflow_datasets as tfds
from dollar_lambda import command


@command()
def run(
    data_dir: Optional[str],
    download_dir: Optional[str],
    context_size: int = 10,
    gamma: float = 0.9,
    max_checkpoint: int = 50,
    test_size: int = 100,
):
    ds_name = "my_dataset"
    builder = tfds.builder(
        ds_name,
        context_size=context_size,
        data_dir=data_dir,
        gamma=gamma,
        max_checkpoint=max_checkpoint,
        test_size=test_size,
    )

    builder.download_and_prepare(download_dir=download_dir)
    df = tfds.load(
        ds_name,
        builder_kwargs=dict(
            context_size=context_size,
            gamma=gamma,
            max_checkpoint=max_checkpoint,
            test_size=test_size,
        ),
        download_and_prepare_kwargs=dict(download_dir=download_dir),
    )

    for example in df["train"].take(1):
        print("\n\nFirst data point;")
        pprint(example)
        print("\n\n")


if __name__ == "__main__":
    run()
