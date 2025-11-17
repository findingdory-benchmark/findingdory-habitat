#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import os
import sys
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch

from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

import findingdory.config
import findingdory.policies
import findingdory.task

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="baseline/ddppo_imagenav",
)
def main(cfg: "DictConfig"):
    cfg = patch_config(cfg)
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


def execute_exp(config: "DictConfig", run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    from habitat_baselines.common.baseline_registry import baseline_registry

    trainer_init = baseline_registry.get_trainer(
        config.habitat_baselines.trainer_name
    )
    assert (
        trainer_init is not None
    ), f"{config.habitat_baselines.trainer_name} is not supported"
    trainer = trainer_init(config)

    if config.habitat_baselines.evaluate:
        trainer.eval()
    else:
        trainer.train()


if __name__ == "__main__":
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.environ["MAGNUM_LOG"] = "quiet"
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    main()
