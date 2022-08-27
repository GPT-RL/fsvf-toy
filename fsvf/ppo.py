import os
import socket
import sys
import time
from pathlib import Path
from shlex import quote
from typing import Any, Mapping, Optional

import jax
import line
import ray
import yaml
from dollar_lambda import CommandTree, argument, flag, nonpositional
from git.repo import Repo
from ppo.lib import train
from ray import tune
from run_logger import RunLogger, create_sweep

tree = CommandTree()
CONFIG_PATH = Path("fsvf/ppo/config.yml")
DEFAULTS_PATH = Path("fsvf/ppo/default.yml")
ALLOW_DIRTY_FLAG = flag("allow_dirty", default=False)
GRAPHQL_ENDPOINT = os.getenv("GRAPHQL_ENDPOINT")


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
        print("Done!")


@tree.command()
def no_log(
    config_path: Path = CONFIG_PATH, disable_jit: bool = False, render: bool = False
):
    assert GRAPHQL_ENDPOINT is not None
    logger = RunLogger(GRAPHQL_ENDPOINT)
    with DEFAULTS_PATH.open() as f:
        with_defaults = yaml.load(f, Loader=yaml.FullLoader)
    with config_path.open() as f:
        without_defaults = yaml.load(f, Loader=yaml.FullLoader)
    with_defaults.update(without_defaults)
    return no_sweep(
        disable_jit=disable_jit, logger=logger, render=render, **with_defaults
    )


@tree.subcommand(parsers=dict(kwargs=nonpositional(argument("name"))))
def log(allow_dirty: bool = False, config_path: Path = CONFIG_PATH, **kwargs):
    repo = Repo(".")
    with config_path.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    kwargs.update(config)
    with DEFAULTS_PATH.open() as f:
        defaults = yaml.load(f, Loader=yaml.FullLoader)
    return _log(
        **kwargs, allow_dirty=allow_dirty, defaults=defaults, repo=repo, sweep_id=None
    )


def _log(
    allow_dirty: bool,
    defaults: dict[str, Any],
    name: str,
    repo: Repo,
    sweep_id: Optional[int],
    **parameters,
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
        dict(parameters=parameters, run_id=logger.run_id, name=name)
    )
    defaults.update(parameters)
    (no_sweep if sweep_id is None else train)(
        **defaults, disable_jit=False, logger=logger, render=False
    )


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
    with config_path.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    assert GRAPHQL_ENDPOINT is not None
    num_settings = sum(1 for _ in param_generator(config))
    sweep_id = create_sweep(
        config=config,
        graphql_endpoint=GRAPHQL_ENDPOINT,
        log_level="INFO",
        name=name,
        project=None,
    )
    with DEFAULTS_PATH.open() as f:
        defaults = yaml.load(f, Loader=yaml.FullLoader)
    config: "dict[str, Any]" = dict(
        defaults=defaults,
        name=name,
        repo=Repo("."),
        sweep_id=sweep_id,
        **kwargs,
        **{
            k: (tune.choice(v) if random_search else tune.grid_search(v))
            if isinstance(v, list)
            else v
            for k, v in config.items()
        },
    )
    ray.init()
    num_gpu = len(jax.devices())
    assert num_gpu >= num_settings
    num_cpu = os.cpu_count()
    assert num_cpu is not None
    analysis = tune.run(
        trainable,
        config=config,
        resources_per_trial=dict(cpu=num_cpu // num_settings, gpu=1),
    )
    print(analysis.stats())


if __name__ == "__main__":
    tree()
