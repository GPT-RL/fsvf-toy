import os
from pathlib import Path

from build_tree import build_tree
from dollar_lambda import flag
from supervised.train import train


def xy():
    yield "hours", "test accuracy"
    for y in [
        "test accuracy",
        "test loss",
        "accuracy",
        "loss",
        "best dev score",
        "steps per second",
    ]:
        yield "step", y


if __name__ == "__main__":
    default_log_level = dict(log_level="INFO")
    tree = build_tree(
        config_path=Path("fsvf/supervised/config.yml"),
        defaults_path=Path("fsvf/supervised/default.yml"),
        log_defaults=dict(**default_log_level, disable_jit=False),
        no_log_defaults=default_log_level,
        no_log_parser=flag("disable_jit", default=False),
        run=train,
        sweep_defaults=dict(**default_log_level, disable_jit=False),
        xy=xy(),
    )
    tree()
