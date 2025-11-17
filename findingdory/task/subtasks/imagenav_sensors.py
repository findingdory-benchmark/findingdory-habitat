#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import numpy as np
import quaternion

from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import DistanceToGoal, NavigationEpisode, Success
from habitat.tasks.nav.object_nav_task import ObjectGoal
from habitat.utils.geometry_utils import (
    angle_between_quaternions,
    quaternion_from_coeff,
)
from findingdory.utils import quat_to_xy_heading
from findingdory.dataset.utils import get_agent_yaw


@registry.register_sensor
class ImageGoalRotationSensor(RGBSensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.
    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "imagegoal_rotation"

    def __init__(
        self, *args: Any, sim: Simulator, config, **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor) and uuid != "third_rgb"
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                f"ImageGoalNav requires one RGB sensor, {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_goal_position = None
        self._current_goal_rotation = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def get_observation(
        self,
        *args: Any,
        observations,
        task, 
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"

        goal_position = task.nav_goal_pos
        goal_rotation = task.nav_goal_rot

        if episode_uniq_id == self._current_episode_id and (
            goal_position == self._current_goal_position
        ).all() and (goal_rotation == self._current_goal_rotation):
            return self._current_image_goal

        self._current_image_goal = self._sim.get_observations_at(
            position=goal_position, rotation=goal_rotation
        )[self._rgb_sensor_uuid]

        self._current_episode_id = episode_uniq_id
        self._current_goal_position = goal_position
        self._current_goal_rotation = goal_rotation

        return self._current_image_goal


@registry.register_measure
class SimpleReward(Measure):
    cls_uuid: str = "simple_reward"

    def __init__(self, *args: Any, sim: Simulator, config, **kwargs: Any):
        super().__init__(**kwargs)
        self._success_reward = config.success_reward
        self._angle_success_reward = config.angle_success_reward
        self._use_dtg_reward = config.use_dtg_reward
        self._use_atg_reward = config.use_atg_reward
        self._use_atg_fix = config.use_atg_fix
        self._atg_reward_distance = config.atg_reward_distance
        self._slack_penalty = config.slack_penalty
        self._previous_dtg: Optional[float] = None
        self._previous_atg: Optional[float] = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self,
        *args: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                DistanceToGoal.cls_uuid,
                TrainSuccess.cls_uuid,
                AngleToGoal.cls_uuid,
                AngleSuccess.cls_uuid,
            ],
        )
        self._metric = None
        self._previous_dtg = None
        self._previous_atg = None
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        # success
        success = task.measurements.measures[TrainSuccess.cls_uuid].get_metric()
        success_reward = self._success_reward if success else 0.0

        # distance-to-goal
        dtg = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        if self._previous_dtg is None:
            self._previous_dtg = dtg

        dtg_reward = self._previous_dtg - dtg if self._use_dtg_reward else 0.0
        self._previous_dtg = dtg

        # angle-to-goal
        atg = task.measurements.measures[AngleToGoal.cls_uuid].get_metric()
        add_atg = self._use_atg_reward
        if self._use_atg_fix:
            if dtg > self._atg_reward_distance:
                atg = np.pi
        else:
            if dtg > self._atg_reward_distance:
                add_atg = False
        if self._previous_atg is None:
            self._previous_atg = atg
        atg_reward = self._previous_atg - atg if add_atg else 0.0
        self._previous_atg = atg

        # angle success
        angle_success = task.measurements.measures[AngleSuccess.cls_uuid].get_metric()
        angle_success_reward = (
            self._angle_success_reward if angle_success else 0.0
        )

        # slack penalty
        slack_penalty = self._slack_penalty

        self._metric = (
            success_reward
            + dtg_reward
            + atg_reward
            + angle_success_reward
            + slack_penalty
        )


@registry.register_measure
class AngleToGoal(Measure):
    """The measure calculates an angle towards the goal. Note: this measure is
    only valid for single goal tasks (e.g., ImageNav)
    """
    cls_uuid: str = "angle_to_goal"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        super().__init__()
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = None
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode: NavigationEpisode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        current_rotation = get_agent_yaw(self._sim.get_agent_state())
        if not isinstance(current_rotation, quaternion.quaternion):
            if isinstance(current_rotation, float) or isinstance(current_rotation, np.float32):
                current_rotation = [0, np.sin(current_rotation / 2), 0, np.cos(current_rotation / 2)]

            current_rotation = quaternion_from_coeff(current_rotation)

        assert task.nav_goal_rot is not None, "Episode must have goals"

        goal_rotation = task.nav_goal_rot
        if not isinstance(goal_rotation, quaternion.quaternion):
            if isinstance(goal_rotation, float):
                goal_rotation = [0, np.sin(goal_rotation / 2), 0, np.cos(goal_rotation / 2)]

            goal_rotation = quaternion_from_coeff(goal_rotation)

        self._metric = angle_between_quaternions(current_rotation, goal_rotation)


@registry.register_measure
class AngleSuccess(Measure):
    """Weather or not the agent is within an angle tolerance."""

    cls_uuid: str = "angle_success"

    def __init__(self, config, *args: Any, **kwargs: Any):
        self._use_train_success = config.use_train_success
        self._success_angle = config.success_angle

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        dependencies = [AngleToGoal.cls_uuid]
        if self._use_train_success:
            dependencies.append(TrainSuccess.cls_uuid)
        else:
            dependencies.append(Success.cls_uuid)
        task.measurements.check_measure_dependencies(self.uuid, dependencies)
        self.update_metric(task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        if self._use_train_success:
            success = task.measurements.measures[TrainSuccess.cls_uuid].get_metric()
        else:
            success = task.measurements.measures[Success.cls_uuid].get_metric()
        angle_to_goal = task.measurements.measures[AngleToGoal.cls_uuid].get_metric()

        if success and np.rad2deg(angle_to_goal) < self._success_angle:
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure
class TrainSuccess(Success):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """
    cls_uuid: str = "train_success"
    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        if (
            hasattr(task, "is_stop_called")
            and task.is_stop_called  # type: ignore
            and distance_to_target < self._success_distance
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0