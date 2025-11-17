#!/usr/bin/env python3

from collections import defaultdict
import os
import shutil
from typing import Dict, Optional
from tqdm import tqdm

import numpy as np

from habitat.config.default import get_config
from habitat.core.env import Env

from findingdory.dataset.utils import (
    create_final_image,
    save_mp4,
    MetaDataSaver,
)
import findingdory.task

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
np.random.seed(31)

VERSION = "v1"

class DataCollector:
    r"""DataCollector for agents in Habitat environments."""

    def __init__(self, env, output_folder: str) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        """
        self._env = env
        self._output_folder = output_folder
        if not os.path.exists(self._output_folder):
            os.makedirs(self._output_folder)
        self.obj_category_id_to_obj_category = {
            v: k for k, v in self._env._dataset.obj_category_to_obj_category_id.items()
        }
        self.recep_category_id_to_recep_category = {
            v: k
            for k, v in self._env._dataset.recep_category_to_recep_category_id.items()
        }
        self.all_obj_category_id_to_obj_category = {
            v: k for k, v in self._env._dataset.other_obj_category_to_other_obj_category_id.items()
        }
        self.object_ids_start = self._env._sim.habitat_config.object_ids_start

        self.metadata_saver = MetaDataSaver()
        self.reset()

        
    def _get_category_id_to_instance_ids(self):
        # use self._env.task.object_semantic_ids
        if getattr(self._env.task, "object_semantic_ids", None) is not None:
            for instance_id, category_id in self._env.task.object_semantic_ids.items():
                if category_id not in self.obj_category_id_to_instance_ids:
                    self.obj_category_id_to_instance_ids[category_id] = []
                self.obj_category_id_to_instance_ids[category_id].append(instance_id)

        # use self._env.task.receptacle_semantic_ids
        if getattr(self._env.task, "receptacle_semantic_ids", None) is not None:
            for instance_id, category_id in self._env.task.receptacle_semantic_ids.items():
                if category_id not in self.recep_category_id_to_instance_ids:
                    self.recep_category_id_to_instance_ids[category_id] = []
                self.recep_category_id_to_instance_ids[category_id].append(instance_id)

        # use self._env.task.other_object_semantic_ids
        if getattr(self._env.task, "other_object_semantic_ids", None) is not None:
            for instance_id, category_id in self._env.task.other_object_semantic_ids.items():
                if category_id not in self.all_obj_category_id_to_instance_ids:
                    self.all_obj_category_id_to_instance_ids[category_id] = []
                self.all_obj_category_id_to_instance_ids[category_id].append(instance_id)
            
    def _update_oracle_goal_details(self):       
        # Extract the pick goal and place goal details from the current oracle agent goal
        oracle_goal = self._env.task._current_oracle_goal

        pick_goal_name = oracle_goal['pick_goal'].object_category
        place_goal_name = oracle_goal['place_goal'].object_category

        print("Selecting candidate object for pick: {} and place {}".format(pick_goal_name, place_goal_name))
        
        return pick_goal_name, place_goal_name

    def evaluate(self, num_episodes: Optional[int] = None) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if num_episodes == -1:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(num_episodes, len(self._env.episodes))
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        pbar = tqdm(total=num_episodes)
        while count_episodes < num_episodes:
            observations = self._env.reset()
            self.reset()

            goal_names = []

            self.update_metadata(observations)

            print("--" * 20, "Running Episode: ", self._env.current_episode.episode_id)

            pbar1 = tqdm(total=self._env._max_episode_steps)
            while not self._env.episode_over and self._env.task.oracle_agent.assert_counter < 10:

                pick_goal_name, place_goal_name = self._update_oracle_goal_details()
                action_name = self._env.task.oracle_agent.get_current_action_name

                self.update_metadata(
                    observations,
                    action_name
                )

                assert place_goal_name is not None, "Place Goal is None"
                goal_names.append(pick_goal_name)
                goal_names.append(place_goal_name)
                
                prev_oracle_goal_idx = self._env.task._current_oracle_goal_idx
                
                # Perform data collection in current episode
                while not self._env.episode_over:
                    observations = self._env.step({'action': 0})    # Pass STOP action so that episode ends as soon as data_collection phase is over
                    
                    if not self._env.task._data_collection_phase:
                        break
                    
                    action_name = self._env.task.oracle_agent.get_current_action_name
                    
                    if self._env.task._current_oracle_goal_idx != prev_oracle_goal_idx:
                        pick_goal_name, place_goal_name = self._update_oracle_goal_details()
                        prev_oracle_goal_idx = self._env.task._current_oracle_goal_idx
                        goal_names.append(pick_goal_name)
                        goal_names.append(place_goal_name)

                    self.update_metadata(observations, action_name)
                    pbar1.update(1)
                
                break

            # self.metadata_saver.save(folder)
            self.save_video(f"{count_episodes:04d}" + "_" + "_".join(goal_names))

            metrics = self._env.get_metrics()
            for m, v in metrics.items():
                if m == "top_down_map":
                    continue
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v
            count_episodes += 1
            pbar.update(1)

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics

    def save_video(self, video_name):
        save_mp4(self._images, os.path.join(self._output_folder, f"{video_name}"))

    def update_metadata(self, observations, sem_map_result=None, action=""):
        # get room annotation
        robot_in_regions = self._env._sim.semantic_scene.get_regions_for_points(
            [self._env._sim.get_agent_state().position]
        )
        robot_region_ids = []
        for tup in robot_in_regions:
            robot_region_ids.append(self._env._sim.semantic_scene.regions[tup[0]].id)

        classes_in_frame, recep_classes_in_frame, other_classes_in_frame = self._get_objects_in_frame(observations)
        time_of_day = observations.get("time_of_day", None)
        robot_pos = np.concatenate([observations.get("robot_start_gps", np.array([0.0, 0.0])), observations.get("robot_start_compass", np.array([0.0]))*180/np.pi])
        goal_name = observations.get("lang_memory_goal", None)
        
        if sem_map_result is not None:
            sem_map_ego_frame = sem_map_result[0]
            sem_map_vis = sem_map_result[1]
            obstacle_map_vis = sem_map_result[2]
        else:
            sem_map_ego_frame = None
            sem_map_vis = None
            obstacle_map_vis = None

        self._images.append(
            create_final_image(
                observations,
                goal_name,
                self._env.get_metrics().get("top_down_map", None),
                classes_in_frame,
                recep_classes_in_frame,
                other_classes_in_frame,
                robot_region_ids,
                sem_map_ego_frame,
                sem_map_vis,
                obstacle_map_vis,
                time_of_day,
                action,
                robot_pos,
            )
        )
        self.metadata_saver.add(
            len(self._images),
            goal_name,
            classes_in_frame,
            recep_classes_in_frame,
            other_classes_in_frame,
            robot_region_ids,
            time_of_day,
            action,
            robot_pos,
        )

    def _get_objects_in_frame(self, observations):
        object_segment = observations.get("all_object_segmentation", [])
        recep_segment = observations.get("receptacle_segmentation", [])
        other_obect_segment = observations.get("other_object_segmentation", [])
        panoptic_segment = observations.get("head_panoptic", [])
        all_object_segmentation = np.unique(object_segment)
        all_receptacle_segmentation = np.unique(recep_segment)
        all_other_object_segmentation = np.unique(other_obect_segment)
        all_panoptic_segmentation = np.unique(panoptic_segment)

        objects_in_frame, receps_in_frame, other_objects_in_frame = [], [], []
        # Got all images, now write information on the image
        # object category details
        for object_class_id in all_object_segmentation:
            if object_class_id != 0:
                obj_category = self.obj_category_id_to_obj_category[object_class_id - 1]
                # the task adds +1 to the category id
                obj_instance_ids = np.array(self.obj_category_id_to_instance_ids[object_class_id]) + self.object_ids_start
                for obj_instance_id in obj_instance_ids:
                    if obj_instance_id in all_panoptic_segmentation:
                        # get the index of the object in the panoptic segmentation
                        panoptic_index = np.where(obj_instance_id == obj_instance_ids)
                        objects_in_frame.append(obj_category + "_" + str(panoptic_index[0][0]))

        # receptacle category details
        for receptacle_class_id in all_receptacle_segmentation:
            if receptacle_class_id != 0:
                recep_category = self.recep_category_id_to_recep_category[receptacle_class_id - 1]
                recep_instance_ids = np.array(self.recep_category_id_to_instance_ids[receptacle_class_id]) + self.object_ids_start
                for recep_instance_id in recep_instance_ids:
                    if recep_instance_id in all_panoptic_segmentation:
                        # get the index of the object in the panoptic segmentation
                        panoptic_index = np.where(recep_instance_id == recep_instance_ids)
                        receps_in_frame.append(recep_category + "_" + str(panoptic_index[0][0]))

        # other object category details
        for other_object_class_id in all_other_object_segmentation:
            if other_object_class_id != 0:
                other_obj_category = self.all_obj_category_id_to_obj_category[other_object_class_id - 1]
                if other_obj_category == "":
                    continue

                other_obj_instance_ids = np.array(self.all_obj_category_id_to_instance_ids[other_object_class_id]) + self.object_ids_start
                for other_obj_instance_id in other_obj_instance_ids:
                    if other_obj_instance_id in all_panoptic_segmentation:
                        # get the index of the object in the panoptic segmentation
                        panoptic_index = np.where(other_obj_instance_id == other_obj_instance_ids)
                        other_objects_in_frame.append(other_obj_category + "_" + str(panoptic_index[0][0]))

        return objects_in_frame, receps_in_frame, other_objects_in_frame

    def reset(self):
        self.metadata_saver.clear()
        self.obj_category_id_to_instance_ids = {}
        self.recep_category_id_to_instance_ids = {}
        self.all_obj_category_id_to_instance_ids = {}
        self._images = []
        self._get_category_id_to_instance_ids()


def main():

    directory_path = "debug/data_collector/"

    env = Env(config=get_config("config/benchmark/findingdory.yaml"))

    data_collector = DataCollector(
        env,
        directory_path,
    )
    
    if os.path.exists(directory_path):
        # Delete the existing directory
        shutil.rmtree(directory_path)

    # Create the new directory
    os.makedirs(directory_path)

    metrics = data_collector.evaluate(-1)
    print(metrics)


if __name__ == "__main__":
    main()