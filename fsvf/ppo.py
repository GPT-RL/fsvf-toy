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
from pathlib import Path

from build_tree import build_tree
from dollar_lambda import flag
from ppo.lib import train


def xy():
    for x in ["frames", "hours", "steps"]:
        yield x, "return"
    yield "steps", "save count"


if __name__ == "__main__":
    download_dir = os.getenv("DOWNLOAD_DIR")
    assert download_dir is not None
    render_default = dict(render=False)
    download_dir_default = dict(download_dir=Path(download_dir))
    tree = build_tree(
        config_path=Path("fsvf/ppo/config.yml"),
        defaults_path=Path("fsvf/ppo/default.yml"),
        log_defaults=dict(**download_dir_default, disable_jit=False, **render_default),
        no_log_defaults=dict(download_dir=None, **render_default),
        no_log_parser=flag("disable_jit", default=False),
        run=train,
        sweep_defaults=dict(
            disable_jit=False, **download_dir_default, **render_default
        ),
        xy=xy(),
    )
    tree()
