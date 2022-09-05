import os
from pathlib import Path
from typing import Optional

import line
import yaml
from build_tree import GRAPHQL_ENDPOINT, build_tree, param_generator
from dollar_lambda import argument, flag
from run_logger import RunLogger, get_load_params
from supervised.train import train


def xy():
    yield "hours", "test error"
    for y in [
        "test error",
        "test loss",
        "test round 1 accuracy",
        "test round 2 accuracy",
        "error",
        "loss",
        "round 1 accuracy",
        "round 2 accuracy",
        "best dev score",
        "steps per second",
        "save count",
    ]:
        yield "step", y


if __name__ == "__main__":
    visualizer_url = os.getenv("VISUALIZER_URL")
    assert visualizer_url is not None, "VISUALIZER_URL must be set"

    defaults = dict(log_level="INFO", load_path=None)
    defaults_path = Path("fsvf/supervised/default.yml")
    tree = build_tree(
        charts=[
            line.spec(
                color="run ID",
                x=x,
                y=y,
                scale_type="log" if ("loss" in y) or ("error" in y) else "linear",
                visualizer_url=visualizer_url,
            )
            for x, y in xy()
        ],
        config_path=Path("fsvf/supervised/config.yml"),
        defaults_path=defaults_path,
        log_defaults=dict(**defaults, disable_jit=False),
        no_log_defaults=defaults,
        no_log_parser=flag("disable_jit", default=False),
        run=train,
        sweep_defaults=dict(**defaults, disable_jit=False),
    )

    @tree.subcommand(parsers=dict(load_id=argument("load_id", type=int)))
    def load(load_id: int, disable_jit: bool = False, load_dir: Optional[str] = None):
        assert GRAPHQL_ENDPOINT is not None
        logger = RunLogger(GRAPHQL_ENDPOINT)
        params = get_load_params(load_id, logger)
        if load_dir is None:
            with Path("fsvf/supervised/config.yml").open() as f:
                load_dir_str = yaml.load(f, Loader=yaml.FullLoader)["save_dir"]
            assert load_dir_str is not None
            load_dir = Path(load_dir_str)
        load_path = Path(load_dir, str(load_id))
        with defaults_path.open() as f:
            with_defaults = yaml.load(f, Loader=yaml.FullLoader)
        with_defaults.update(params, disable_jit=disable_jit)
        params = next(param_generator(with_defaults))
        return train(**params, load_path=load_path, log_level="INFO", run_logger=logger)

    tree()
