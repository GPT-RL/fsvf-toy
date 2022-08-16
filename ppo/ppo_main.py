# Copyright 2022 The Flax Authors.
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

# See issue #620.
# pytype: disable=wrong-keyword-args

import os
import socket
import sys
import time
from pathlib import Path
from shlex import quote
from typing import Optional

import line
import models
import tensorflow as tf
import yaml
from dollar_lambda import CommandTree, argument, flag, nonpositional
from git import Repo
from gym_minigrid.minigrid import MiniGridEnv
from ppo_lib import train
from run_logger import HasuraLogger

tree = CommandTree()
DEFAULT_CONFIG = Path("config.yml")
GRAPHQL_ENDPOINT = os.getenv("GRAPHQL_ENDPOINT")
ALLOW_DIRTY_FLAG = flag("allow_dirty", default=False)


def main(**kwargs):
    num_actions = len(MiniGridEnv.Actions)
    model = models.TwoLayer(num_outputs=num_actions)
    return train(model=model, **kwargs)


@tree.command()
def no_log(config_path: Path = DEFAULT_CONFIG):
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], "GPU")
    with config_path.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger = HasuraLogger(GRAPHQL_ENDPOINT)
    return main(**config, logger=logger)


@tree.subcommand(parsers=dict(kwargs=nonpositional(argument("name"))))
def log(allow_dirty: bool = False, config_path: Path = DEFAULT_CONFIG, **kwargs):
    repo = Repo(".")
    with config_path.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config.update(kwargs)
    return _log(**config, allow_dirty=allow_dirty, repo=repo, sweep_id=None)


def _log(
    allow_dirty: bool,
    name: str,
    repo: Repo,
    sweep_id: Optional[int],
    **kwargs,
):
    if not allow_dirty:
        assert not repo.is_dirty()

    metadata = dict(
        reproducibility=(
            dict(
                command_line=f'python {" ".join(quote(arg) for arg in sys.argv)}',
                time=time.strftime("%c"),
                cwd=str(Path.cwd()),
                commit=str(repo.commit()),
                remotes=[*repo.remote().urls],
            )
        ),
        hostname=socket.gethostname(),
    )

    visualizer_url = os.getenv("VISUALIZER_URL")
    assert visualizer_url is not None, "VISUALIZER_URL must be set"

    def xy():
        for x in ["step", "hours"]:
            yield x, "return"

    charts = [
        line.spec(color="seed", x=x, y=y, visualizer_url=visualizer_url)
        for x, y in xy()
    ]

    logger = HasuraLogger(GRAPHQL_ENDPOINT)
    logger.create_run(metadata=metadata, sweep_id=sweep_id, charts=charts)
    logger.update_metadata(  # this updates the metadata stored in the database
        dict(
            parameters=kwargs,
            run_id=logger.run_id,
            name=name,
        )
    )  # todo: encapsulate in HasuraLogger
    main(**kwargs, logger=logger)


if __name__ == "__main__":
    tree()
