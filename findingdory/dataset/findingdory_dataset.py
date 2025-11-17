#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, cast

import attr
import numpy as np

from habitat.core.dataset import EpisodeIterator
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.datasets.ovmm.ovmm_dataset import OVMMDatasetV0
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation
from habitat.tasks.nav.nav import NavigationGoal

@dataclass
class ReceptacleSampleInfo:
    type: List[str]
    attribute: Optional[str] = None

@dataclass
class ObjectSampleInfo:
    type: List[str]
    attribute: Optional[str] = None


@dataclass
class InstructionInfo:
    task_id: str
    lang: str
    task_type: str
    goal_expr: Any
    sampled_objects: List[ObjectSampleInfo]
    sampled_receps: List[ReceptacleSampleInfo]
    sequential_goals: Optional[Dict] = None
    temporal_target_entity: str = None # NOTE: THIS SHOULD NOT BE MERGED. Added it temporarily to avoid errors with loading of training dataset for imagenav


@attr.s(auto_attribs=True, kw_only=True)
class FindingDoryEpisode(RearrangeEpisode):
    r"""Specifies categories of the object, start and goal receptacles

    :property object_category: Category of the object to be rearranged
    :property start_recep_category: Category of the start receptacle
    :property goal_recep_category: Category of the goal receptacle
    """
    object_category: Optional[str] = None
    start_recep_category: Optional[str] = None
    goal_recep_category: Optional[str] = None
    candidate_objects: Optional[List[ObjectGoal]] = None
    candidate_objects_noninteracted: Optional[List[ObjectGoal]] = None
    candidate_start_receps: Optional[List[ObjectGoal]] = None
    candidate_start_receps_noninteracted: Optional[List[ObjectGoal]] = None
    candidate_goal_receps: Optional[List[ObjectGoal]] = None
    candidate_goal_receps_noninteracted: Optional[List[ObjectGoal]] = None
    instructions: Optional[List[InstructionInfo]] = None
    place_valid_receps: Optional[Dict] = None
    place_oracle_action_seq: Optional[Dict] = None
    nav_goal_pos: Optional[NavigationGoal] = None
    nav_goal_rot: Optional[np.ndarray] = None


class FindingDoryEpisodeIterator(EpisodeIterator[FindingDoryEpisode]):
    def __init__(
        self,
        viewpoints_matrix,
        transformations_matrix,
        episodes,
        *args,
        **kwargs
    ):
        self.viewpoints = viewpoints_matrix
        self.transformations = transformations_matrix
        self._vp_keys = [
            "candidate_objects",
            "candidate_objects_noninteracted",
            "candidate_start_receps",
            "candidate_start_receps_noninteracted",
            "candidate_goal_receps",
            "candidate_goal_receps_noninteracted",
        ]
        super().__init__(episodes, *args, **kwargs)

    def __next__(self) -> FindingDoryEpisode:
        # deepcopy is to avoid increasing memory as we iterate through the episodes
        episode = cast(FindingDoryEpisode, copy.deepcopy(super().__next__()))

        instructions: List[InstructionInfo] = []
        if episode.instructions is not None:
            for instruction in episode.instructions.values():
                instructions.append(InstructionInfo(**instruction))
            episode.instructions = instructions

        deserialized_objs = []
        if self.transformations is not None:
            for rigid_obj in episode.rigid_objs:
                transform = np.vstack(
                    (self.transformations[rigid_obj[1]], [0, 0, 0, 1])
                )
                deserialized_objs.append((rigid_obj[0], transform))
            episode.rigid_objs = deserialized_objs

        if self.viewpoints is None or self.viewpoints.size == 0:
            return episode

        for vp_key in self._vp_keys:
            obj_goal: ObjectGoal
            for obj_goal in getattr(episode, vp_key):
                for vidx, view_idx in enumerate(obj_goal.view_points):
                    view = self.viewpoints[view_idx]
                    position, rotation, iou = (
                        view[:3],
                        view[3:7],
                        view[7].item(),
                    )
                    agent_state = AgentState(position, rotation)
                    obj_goal.view_points[vidx] = ObjectViewLocation(
                        agent_state, iou
                    )

        return episode


@registry.register_dataset(name="FindingDoryDataset-v0")
class FindingDoryDatasetV0(OVMMDatasetV0):
    r"""Class inherited from OVMMDataset."""
    other_obj_category_to_other_obj_category_id: Dict[str, int]

    def get_episode_iterator(
        self, *args: Any, **kwargs: Any
    ) -> FindingDoryEpisodeIterator:
        return FindingDoryEpisodeIterator(
            self.viewpoints_matrix,
            self.transformations_matrix,
            self.episodes,
            *args,
            **kwargs
        )

    def __deserialize_goal(
        self, serialized_goal: Dict[str, Any]
    ) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)
        if self.viewpoints_matrix is None:
            # if the view points are not cached separately, read from original episodes
            for vidx, view in enumerate(g.view_points):
                view_location = ObjectViewLocation(**view)  # type: ignore
                view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
                g.view_points[vidx] = view_location
        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)

        if "obj_category_to_obj_category_id" in deserialized:
            self.obj_category_to_obj_category_id = deserialized[
                "obj_category_to_obj_category_id"
            ]
        if "recep_category_to_recep_category_id" in deserialized:
            self.recep_category_to_recep_category_id = deserialized[
                "recep_category_to_recep_category_id"
            ]
        if "other_obj_category_to_other_obj_category_id" in deserialized:
            self.other_obj_category_to_other_obj_category_id = deserialized[
                "other_obj_category_to_other_obj_category_id"
            ]

        all_episodes = deserialized["episodes"]
        if self.episode_indices_range is None:
            episodes_index_low, episodes_index_high = 0, len(all_episodes)
        else:
            (
                episodes_index_low,
                episodes_index_high,
            ) = self.episode_indices_range

        episode_ids_subset = None
        if len(self.config.episode_ids) > 0:
            episode_ids_subset = self.config.episode_ids[
                episodes_index_low:episodes_index_high
            ]
        else:
            all_episodes = all_episodes[episodes_index_low:episodes_index_high]

        for episode in all_episodes:
            rearrangement_episode = FindingDoryEpisode(**episode)
            for goal_type in [
                "candidate_objects",
                "candidate_objects_noninteracted",
                "candidate_start_receps",
                "candidate_start_receps_noninteracted",
                "candidate_goal_receps",
                "candidate_goal_receps_noninteracted",
            ]:
                if goal_type in episode:
                    setattr(
                        rearrangement_episode,
                        goal_type,
                        [
                            self.__deserialize_goal(g)
                            for g in episode[goal_type]
                        ],
                    )

            if (
                episode_ids_subset is None
                or int(episode["episode_id"]) in episode_ids_subset
            ):
                self.episodes.append(rearrangement_episode)