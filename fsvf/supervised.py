import os
from pathlib import Path

import line
from build_tree import build_tree
from dollar_lambda import flag
from supervised.train import train


def xy():
    yield "hours", "test error"
    for y in [
        "order accuracy",
        "argmax accuracy",
        "test error",
        "test loss",
        "error",
        "loss",
        "best dev score",
        "steps per second",
    ]:
        yield "step", y


if __name__ == "__main__":
    visualizer_url = os.getenv("VISUALIZER_URL")
    assert visualizer_url is not None, "VISUALIZER_URL must be set"

    default_log_level = dict(log_level="INFO")
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
        defaults_path=Path("fsvf/supervised/default.yml"),
        log_defaults=dict(**default_log_level, disable_jit=False),
        no_log_defaults=default_log_level,
        no_log_parser=flag("disable_jit", default=False),
        run=train,
        sweep_defaults=dict(**default_log_level, disable_jit=False),
    )
    tree()
