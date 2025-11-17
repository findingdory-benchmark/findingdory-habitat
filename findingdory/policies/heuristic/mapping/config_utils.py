# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
from typing import Optional, Tuple

import habitat.config.default
import yaml
from omegaconf import DictConfig, OmegaConf
from habitat.config import read_write


def get_config(
    habitat_config,
) -> Tuple[DictConfig, str]:
    """Adds the ENVIRONMENT parameters to the habitat config dict
    The ENVIRONMENT params are same as the habitat config but required explicitly by the mapping class

    Arguments:
        habitat_config: habitat config dict
        agent_config_path: path to config that defines semanti cmapper agent parameters
        opts: command line arguments overriding the config
    """

    map_config = habitat_config.habitat.task.semantic_mapper

    with read_write(map_config):
        map_config['ENVIRONMENT'] = {}
    
    sim_sensors = habitat_config.habitat.simulator.agents.main_agent.sim_sensors

    if habitat_config.habitat.simulator.agents.main_agent.articulated_agent_type == 'StretchRobot':
        rgb_sensor = sim_sensors.head_rgb_sensor
        depth_sensor = sim_sensors.head_depth_sensor
        semantic_sensor = sim_sensors.head_panoptic_sensor
    else:
        raise RuntimeError("Semantic Mapping code is tested only with StretchRobot config !")
    
    assert rgb_sensor.height == depth_sensor.height
    if semantic_sensor:
        assert rgb_sensor.height == semantic_sensor.height        

    assert rgb_sensor.width == depth_sensor.width
    if semantic_sensor:
        assert rgb_sensor.width == semantic_sensor.width        

    assert rgb_sensor.position[1] == depth_sensor.position[1]
    if semantic_sensor:
        assert rgb_sensor.position[1] == depth_sensor.position[1] == semantic_sensor.position[1]        

    assert rgb_sensor.hfov == depth_sensor.hfov
    if semantic_sensor:
        assert rgb_sensor.hfov == depth_sensor.hfov == semantic_sensor.hfov        

    with read_write(map_config):
        map_config.ENVIRONMENT.frame_height = rgb_sensor.height
        map_config.ENVIRONMENT.frame_width = rgb_sensor.width
        map_config.ENVIRONMENT.camera_height = rgb_sensor.position[1]
        map_config.ENVIRONMENT.hfov = rgb_sensor.hfov
        map_config.ENVIRONMENT.turn_angle = habitat_config.habitat.simulator.turn_angle
        map_config.ENVIRONMENT.min_depth = depth_sensor.min_depth
        map_config.ENVIRONMENT.max_depth = depth_sensor.max_depth
    
    with read_write(habitat_config):
        habitat_config.habitat.task.semantic_mapper = map_config

    return habitat_config