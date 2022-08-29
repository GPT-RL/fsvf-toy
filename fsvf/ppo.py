import os
from pathlib import Path

import line
from build_tree import build_tree
from dollar_lambda import flag
from ppo.lib import train


def xy():
    for x in ["frames", "hours", "steps"]:
        yield x, "return"
    yield "steps", "save count"


if __name__ == "__main__":
    visualizer_url = os.getenv("VISUALIZER_URL")
    assert visualizer_url is not None, "VISUALIZER_URL must be set"
    download_dir = os.getenv("DOWNLOAD_DIR")
    assert download_dir is not None
    render_default = dict(render=False)
    download_dir_default = dict(download_dir=Path(download_dir))
    tree = build_tree(
        charts=[
            line.spec(
                color="run ID",
                x=x,
                y=y,
                scale_type="log" if "loss" in y else "linear",
                visualizer_url=visualizer_url,
            )
            for x, y in xy()
        ],
        config_path=Path("fsvf/ppo/config.yml"),
        defaults_path=Path("fsvf/ppo/default.yml"),
        log_defaults=dict(**download_dir_default, disable_jit=False, **render_default),
        no_log_defaults=dict(download_dir=None, **render_default),
        no_log_parser=flag("disable_jit", default=False),
        run=train,
        sweep_defaults=dict(
            disable_jit=False, **download_dir_default, **render_default
        ),
    )
    tree()
