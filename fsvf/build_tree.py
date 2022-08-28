import os
import socket
import sys
import time
from pathlib import Path
from shlex import quote
from typing import Any, Callable, Iterable, Mapping, Optional

import line
import ray
import yaml
from dollar_lambda import CommandTree, Parser, argument, flag, nonpositional
from git.repo import Repo
from ray import tune
from run_logger import RunLogger, create_sweep


def build_tree(
    config_path: Path,
    run: Callable,
    xy: Iterable[tuple[str, str]],
    defaults_path: Optional[Path] = None,
    log_defaults: Optional[Mapping[str, Any]] = None,
    log_parser: Optional[Parser] = None,
    no_log_defaults: Optional[Mapping[str, Any]] = None,
    no_log_parser: Optional[Parser] = None,
    sweep_defaults: Optional[Mapping[str, Any]] = None,
    sweep_parser: Optional[Parser] = None,
):
    tree = CommandTree()
    GRAPHQL_ENDPOINT = os.getenv("GRAPHQL_ENDPOINT")

    if defaults_path is None:
        defaults = {}
    else:
        with defaults_path.open() as f:
            defaults = yaml.load(f, Loader=yaml.FullLoader)

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
            run(**params)
            print("Done!")

    @tree.command(parsers=dict(kwargs=no_log_parser or Parser.empty()))
    def no_log(config_path: Path = config_path, **kwargs):
        assert GRAPHQL_ENDPOINT is not None
        logger = RunLogger(GRAPHQL_ENDPOINT)
        with_defaults = dict(**defaults)
        with config_path.open() as f:
            without_defaults = yaml.load(f, Loader=yaml.FullLoader)
        with_defaults.update(without_defaults)
        return no_sweep(
            **kwargs, **(no_log_defaults or {}), run_logger=logger, **with_defaults
        )

    kwargs_parser = argument("name")
    if log_parser:
        kwargs_parser = nonpositional(kwargs_parser, log_parser)

    @tree.subcommand(parsers=dict(kwargs=kwargs_parser))
    def log(allow_dirty: bool = False, config_path: Path = config_path, **kwargs):
        repo = Repo(".")
        with config_path.open() as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        kwargs.update(config)
        return _log(
            **kwargs,
            **(log_defaults or {}),
            allow_dirty=allow_dirty,
            defaults=defaults,
            repo=repo,
            sweep_id=None,
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

        charts = [
            line.spec(color="run ID", x=x, y=y, visualizer_url=visualizer_url)
            for x, y in xy
        ]

        assert GRAPHQL_ENDPOINT is not None
        logger = RunLogger(GRAPHQL_ENDPOINT)
        logger.create_run(metadata=metadata, sweep_id=sweep_id, charts=charts)
        logger.update_metadata(  # this updates the metadata stored in the database
            dict(parameters=parameters, run_id=logger.run_id, name=name)
        )
        defaults.update(parameters)
        (no_sweep if sweep_id is None else run)(**defaults, run_logger=logger)

    def trainable(config: dict):
        return _log(**config)

    kwargs_parser = flag("allow_dirty", default=False)
    if sweep_parser:
        kwargs_parser = nonpositional(kwargs_parser, sweep_parser)

    @tree.subcommand(parsers=dict(name=argument("name"), kwargs=kwargs_parser))
    def sweep(
        name: str,
        config_path: Path = config_path,
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
        config: "dict[str, Any]" = dict(
            defaults=defaults,
            name=name,
            repo=Repo("."),
            sweep_id=sweep_id,
            **kwargs,
            **(sweep_defaults or {}),
            **{
                k: (tune.choice(v) if random_search else tune.grid_search(v))
                if isinstance(v, list)
                else v
                for k, v in config.items()
            },
        )
        ray.init()
        num_cpu = os.cpu_count()
        assert num_cpu is not None
        analysis = tune.run(
            trainable,
            config=config,
            resources_per_trial=dict(cpu=num_cpu // num_settings, gpu=1),
        )
        print(analysis.stats())

    return tree
