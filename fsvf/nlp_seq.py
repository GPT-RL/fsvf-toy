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
    dev_path = os.getenv("NLP_SEQ_DEV_PATH")
    train_path = os.getenv("NLP_SEQ_TRAIN_PATH")
    assert dev_path is not None
    assert train_path is not None
    defaults = dict(log_level="INFO", dev=Path(dev_path), train=Path(train_path))
    tree = build_tree(
        config_path=Path("fsvf/supervised/config.yml"),
        defaults_path=Path("fsvf/supervised/default.yml"),
        log_defaults=dict(**defaults, disable_jit=False),
        no_log_defaults=defaults,
        no_log_parser=flag("disable_jit", default=False),
        run=train,
        sweep_defaults=dict(**defaults, disable_jit=False),
        xy=xy(),
    )
    tree()
