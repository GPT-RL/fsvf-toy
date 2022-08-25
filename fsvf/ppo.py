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
from typing import Any, Mapping, Optional

import line
import ray
import yaml
from dollar_lambda import CommandTree, argument, flag, nonpositional
from git.repo import Repo
from ppo.lib import GRAPHQL_ENDPOINT, train
from ray import tune
from rich.console import Console
from run_logger import RunLogger, create_sweep

console = Console()
tree = CommandTree()
CONFIG_PATH = Path("fsvf/ppo/config.yml")
DEFAULTS_PATH = Path("fsvf/ppo/default.yml")
ALLOW_DIRTY_FLAG = flag("allow_dirty", default=False)


def load_config(config_path: Path) -> dict[str, Any]:
    with DEFAULTS_PATH.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config_path.exists():
        with config_path.open() as f:
            config.update(yaml.load(f, Loader=yaml.FullLoader))
    else:
        console.log(f"No config file found at {config_path}")
    return config


def param_generator(params: Any):
    if isinstance(params, Mapping):
        if tuple(params.keys()) == ("",):
            yield from param_generator(params[""])
            return
        if not params:
            yield {}
        else:
            (key, value), *params = params.items()
            for choice in param_generator(value):
                for other_choices in param_generator(dict(params)):
                    yield {key: choice, **other_choices}
    elif isinstance(params, (list, tuple)):
        for choices in params:
            yield from param_generator(choices)
    else:
        yield params


def no_sweep(**kwargs):
    for params in param_generator(kwargs):
        train(**params)
        console.log("Done!")


@tree.command()
def no_log(
    config_path: Path = CONFIG_PATH,
    render: bool = False,
):
    kwargs = load_config(config_path)

    assert GRAPHQL_ENDPOINT is not None
    logger = RunLogger(GRAPHQL_ENDPOINT)
    if kwargs["save_frequency"] != 0:
        console.log("Setting save_frequency to 0.")
    kwargs.update(save_frequency=0)
    return no_sweep(
        **kwargs,
        logger=logger,
        render=render,
    )


@tree.subcommand(parsers=dict(kwargs=nonpositional(argument("name"))))
def log(allow_dirty: bool = False, config_path: Path = CONFIG_PATH, **kwargs):
    repo = Repo(".")
    config = load_config(config_path)
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
        yield "step", "save count"

    charts = [
        line.spec(color="run ID", x=x, y=y, visualizer_url=visualizer_url)
        for x, y in xy()
    ]

    assert GRAPHQL_ENDPOINT is not None
    logger = RunLogger(GRAPHQL_ENDPOINT)
    logger.create_run(metadata=metadata, sweep_id=sweep_id, charts=charts)
    logger.update_metadata(  # this updates the metadata stored in the database
        dict(parameters=kwargs, run_id=logger.run_id, name=name)
    )

    (no_sweep if sweep_id is None else train)(**kwargs, logger=logger, render=False)


def trainable(config: dict):
    return _log(**config)


@tree.subcommand(
    parsers=dict(name=argument("name"), kwargs=nonpositional(ALLOW_DIRTY_FLAG))
)
def sweep(
    name: str,
    config_path: Path = CONFIG_PATH,
    random_search: bool = False,
    **kwargs,
):
    partial_config = load_config(config_path)
    config: "dict[str, Any]" = dict(
        name=name,
        repo=Repo("."),
        **kwargs,
        **{
            k: (tune.choice(v) if random_search else tune.grid_search(v))
            if isinstance(v, list)
            else v
            for k, v in partial_config.items()
        },
    )
    assert GRAPHQL_ENDPOINT is not None
    sweep_id = create_sweep(
        config=config_path,
        graphql_endpoint=GRAPHQL_ENDPOINT,
        log_level="INFO",
        name=name,
        project=None,
    )
    config.update(sweep_id=sweep_id)
    ray.init()
    analysis = tune.run(
        trainable, config=config, resources_per_trial=dict(cpu=4, gpu=1)
    )
    print(analysis.stats())


if __name__ == "__main__":
    tree()
