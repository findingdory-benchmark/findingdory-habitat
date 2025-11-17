from dataclasses import dataclass, field
from typing import Dict, Any

from hydra.core.config_store import ConfigStore
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.config_search_path import ConfigSearchPath

from habitat.config.default_structured_configs import (
    register_hydra_plugin, ActionConfig, ArmActionConfig, LabSensorConfig, MeasurementConfig, TaskConfig, OVMMDatasetConfig, SPLMeasurementConfig
)
from habitat_baselines.config.default_structured_configs import PolicyConfig, RLConfig, HabitatBaselinesRLConfig, VectorEnvFactoryConfig, PPOConfig

cs = ConfigStore.instance()

##########################################################################
# Actions
##########################################################################

@dataclass
class NavigationModeActionConfig(ActionConfig):
    type: str = "NavigationModeAction"
    threshold: float = 0.8

@dataclass
class FindingDoryArmActionConfig(ArmActionConfig):
    type: str = "ArmAction"
    translation_up_offset: float = 0.1
    
@dataclass
class PDDlIntermediateStopActionConfig(ActionConfig):
    type: str = "PddlIntermediateStopAction"

##########################################################################
# Sensors
##########################################################################

@dataclass
class OtherObjectSegmentationSensorConfig(LabSensorConfig):
    type: str = "OtherObjectSegmentationSensor"
    blank_out_prob: float = 0.0

@dataclass
class TimeOfDaySensorConfig(LabSensorConfig):
    type: str = "TimeOfDaySensor"
    wake_up_time: int = 6
    sleep_time: int = 22

@dataclass
class LangMemoryGoalSensorConfig(LabSensorConfig):
    type: str = "LangMemoryGoalSensor"

@dataclass
class LastActionSensorConfig(LabSensorConfig):
    type: str = "LastActionSensor"
    max_action_len: int = 30

@dataclass
class CanTakeActionSensorConfig(LabSensorConfig):
    type: str = "CanTakeActionSensor"

@dataclass
class ImageGoalRotationSensorConfig(LabSensorConfig):
    type: str = "ImageGoalRotationSensor"
    sample_angle: bool = True

@dataclass
class ManipulationModeSensorConfig(LabSensorConfig):
    type: str = "ManipulationModeSensor"

@dataclass
class AgentStateSensorConfig(LabSensorConfig):
    type: str = "AgentStateSensor"
    
@dataclass
class OracleKeyframesSensorConfig(LabSensorConfig):
    type: str = "OracleKeyframesSensor"
    
##########################################################################
# Measures
##########################################################################

@dataclass
class PredicateTaskSuccessConfig(MeasurementConfig):
    type: str = "PredicateTaskSuccess"
    must_call_stop: bool = True
    allows_invalid_actions: bool = True

@dataclass
class SimpleRewardConfig(MeasurementConfig):
    type: str = "SimpleReward"
    success_reward: float = 5.0
    angle_success_reward: float = 5.0
    use_dtg_reward: bool = True
    use_atg_reward: bool = True
    atg_reward_distance: float = 0.8
    slack_penalty: float = -0.002
    use_atg_fix: bool = True

@dataclass
class AngleToGoalConfig(MeasurementConfig):
    type: str = "AngleToGoal"

@dataclass
class AngleSuccessConfig(MeasurementConfig):
    type: str = "AngleSuccess"
    success_angle: float = 25.0
    use_train_success: bool = True

@dataclass
class TrainSuccessConfig(MeasurementConfig):
    type: str = "TrainSuccess"
    success_distance: float = 0.8
    
@dataclass
class PickedObjAnywhereOnGoal(MeasurementConfig):
    type: str = "PickedObjAnywhereOnGoal"
    max_floor_height: float = 0.05

@dataclass
class HighLevelGoalSuccessConfig(MeasurementConfig):
    type: str = "HighLevelGoalSuccess"

@dataclass
class VLMFailureModeSuccessConfig(MeasurementConfig):
    type: str = "VLMFailureModeSuccess"
    
@dataclass
class HighLevelDTGSuccessConfig(MeasurementConfig):
    type: str = "HighLevelDTGSuccess"

@dataclass
class HighLevelSemCovSuccessConfig(MeasurementConfig):
    type: str = "HighLevelSemCovSuccess"
    
@dataclass
class SubGoalCountConfig(MeasurementConfig):
    type: str = "SubGoalCount"
    
@dataclass
class LowLevelPolicyFailureModeConfig(MeasurementConfig):
    type: str = "LowLevelPolicyFailureModeSuccess"
    low_level_policy_max_steps: int = 1000
    relaxed_dtg_thresh: float = 2.5

@dataclass
class FindingDorySPLMeasurementConfig(SPLMeasurementConfig):
    type: str = "FindingDorySPL"

@dataclass
class OracleSolutionGeneratorConfig(MeasurementConfig):
    type: str = "OracleSolutionGenerator"
    result_save_path: str = "debug/vlm_inference_results"
    
@dataclass
class FindingDoryHighLevelSPLConfig(MeasurementConfig):
    type: str = "FindingDoryHighLevelSPL"

##########################################################################
# Task
##########################################################################

@dataclass
class SemanticMapConfig:
    semantic_categories: str = "findingdory_receptacles" # map semantic channel categories ("coco_indoor", "longtail_indoor", "mukul_indoor")
    num_sem_categories: int = 22           # number of map semantic channel categories (16, 257, 35)
    map_size_cm: int = 4800        # global map size (in centimeters)
    map_resolution: int = 5        # size of map bins (in centimeters)
    vision_range: int = 100        # diameter of local map region visible by the agent (in cells)
    global_downscaling: int = 2    # ratio of global over local map
    du_scale: int = 4              # frame downscaling before projecting to point cloud
    cat_pred_threshold: float = 5.0  # number of depth points to be in bin to classify it as a certain semantic category
    exp_pred_threshold: float = 1.0  # number of depth points to be in bin to consider it as explored
    map_pred_threshold: float = 5.0  # number of depth points to be in bin to consider it as obstacle
    explored_radius: int = 50     # radius (in centimeters) of visually explored region
    been_close_to_radius: int = 200  # radius (in centimeters) of been close to region
    must_explore_close: bool = True
    min_obs_height_cm: int = 40    # minimum height (in centimeters) of obstacle to be considered as obstacle
    # erosion and filtering to reduce the number of spurious artifacts
    dilate_obstacles: bool = False
    dilate_size: int = 3
    dilate_iter: int = 1
    # instance mapping
    record_instance_ids: bool = False  # whether to predict and store instance ids in the map
    max_instances: int = 800

@dataclass
class DiscretePlannerConfig:
    collision_threshold: float = 0.20       # forward move distance under which we consider there's a collision (in meters)
    dilate_obstacles: bool = False           # whetehr to dilate obsatcles or not during planning
    min_obs_dilation_selem_radius: int = 3    # radius (in cells) of obstacle dilation structuring element
    obs_dilation_selem_radius: int = 2    # radius (in cells) of obstacle dilation structuring element
    goal_dilation_selem_radius: int = 10  # radius (in cells) of goal dilation structuring element
    use_dilation_for_stg: bool = False      # use dilated goals for estimating short-term goals - or just reaching
    map_downsample_factor: int = 1            # optional downsampling of traversible and goal map before fmm distance call (1 for no downsampling, 2 for halving resolution)
    map_update_frequency: int = 1             # compute fmm distance map every n steps 
    step_size: int = 5                    # maximum distance of the short-term goal selected by the planner
    discrete_actions: bool = True         # discrete motion planner output space or not
    verbose: bool = False
    min_goal_distance_cm: float = 10.0
    goal_tolerance: float = 0.01

@dataclass
class MappingAgentConfig:
    radius: float = 0.05            # robot radius (in meters)
    store_all_categories: bool = True  # whether to store all semantic categories in the map or just task-relevant ones
    verbose: bool = False
    exploration_strategy: str = "seen_frontier"  # exploration strategy ("seen_frontier", "been_close_to_frontier")
    max_steps: int = 100000          # maximum number of steps before stopping an episode

    SEMANTIC_MAP: SemanticMapConfig = SemanticMapConfig()
    PLANNER: DiscretePlannerConfig = DiscretePlannerConfig()

@dataclass
class SemanticMappingConfig:

    NUM_ENVIRONMENTS: int = 1      # number of environments (per agent process)
    DUMP_LOCATION: str = "datadump/hssd_sem_mapper_datadump"   # path to dump models and log
    EXP_NAME: str = "debug"       # experiment name
    VISUALIZE: int = 0              # 1: render observation and predicted semantic map, 0: no visualization
    PRINT_IMAGES: int = 1           # 1: save visualization as images, 0: no image saving
    GROUND_TRUTH_SEMANTICS: int = 1 # 1: use ground-truth semantics (for debugging / ablations)
    seed: int = 0
    SHOW_RL_OBS: bool = False         # whether to show the observations passed to RL policices, for debugging

    AGENT: MappingAgentConfig = MappingAgentConfig()

@dataclass
class FindingDoryTaskConfig(TaskConfig):
    other_object_categories_file: str = ""
    pddl_domain_def_path: str = "config/task/pddl_domain_fp"
    num_steps_daily: int = 1000
    selected_instruction: int = -1
    object_attributes_file_path: str = "data/objects/object_attribute_annotation.csv"
    room_objects_file_path: str = "data/hssd-hab/metadata/room_objects.json"
    instructions_to_evaluate: list = field(default_factory=lambda: [-1])
    enable_semantic_mapping: bool = False
    semantic_mapper: SemanticMappingConfig = SemanticMappingConfig()

@dataclass
class ImageNavTaskConfig(TaskConfig):
    camera_tilt: float = -0.5236
    object_in_hand_sample_prob: float = 0.0
    min_start_distance: float = 1.0

@dataclass
class FindingDoryDatasetConfig(OVMMDatasetConfig):
    type: str = "FindingDoryDataset-v0"

##########################################################################
# Task
##########################################################################
@dataclass
class VisualEncoderTransformsConfig:
    _target_: str = "findingdory.trainer.image_transforms.transform_augment"
    resize_size: list = field(default_factory=lambda: [160, 120])
    output_size: list = field(default_factory=lambda: [160, 120])
    jitter: bool = True
    jitter_prob: float = 1.0
    jitter_brightness: float = 0.3
    jitter_contrast: float = 0.3
    jitter_saturation: float = 0.3
    jitter_hue: float = 0.3
    shift: bool = True
    shift_pad: int = 4
    randomize_environments: bool = False

@dataclass
class VisualEncoderModelConfig:
    _target_: str = "findingdory.policies.end_to_end.vit.vit_base_patch16"
    img_size: list = field(default_factory=lambda: [160, 120])
    use_cls: bool = False
    global_pool: bool = False
    drop_path_rate: float = 0.0

@dataclass
class VisualEncoderConfig:
    _target_: str = "vc_models.models.vit.vit.load_mae_encoder"
    checkpoint_path: str = "vc_models/vc1_vitb.pth"
    model: VisualEncoderModelConfig = VisualEncoderModelConfig()

@dataclass
class VisualEncoderMetadataConfig:
    algo: str = "mae"
    model: str = "vit_base_patch16"
    data: list = field(default_factory=lambda: ["ego", "imagenet", "inav"])
    comment: str = "182_epochs"

@dataclass
class BackboneConfig:
    _target_: str = "vc_models.models.load_model"
    model: VisualEncoderConfig = VisualEncoderConfig()
    transform: VisualEncoderTransformsConfig = VisualEncoderTransformsConfig()
    metadata: VisualEncoderMetadataConfig = VisualEncoderMetadataConfig()


@dataclass
class ImageNavPolicyConfig(PolicyConfig):
    name: str = "ImageNavPolicy"
    backbone_config: BackboneConfig = BackboneConfig()
    rnn_type: str = "GRU"
    num_recurrent_layers: int = 1
    use_augmentations: bool = False
    use_augmentations_test_time: bool = False
    normalize_visual_inputs: bool = False
    freeze_backbone: bool = False
    
@dataclass
class ImageNavPPOConfig(PPOConfig):
    encoder_lr: float = 2.5e-4
    wd: float = 1e-6


@dataclass
class ImageNavRLConfig(RLConfig):
    ppo: ImageNavPPOConfig = ImageNavPPOConfig()
    policy: Dict[str, Any] = field(
        default_factory=lambda: {"main_agent": ImageNavPolicyConfig()}
    )


@dataclass
class ImageNavVectorEnvFactoryConfig(VectorEnvFactoryConfig):
    _target_: str = "findingdory.task.subtasks.imagenav_env.ImageNavVectorEnvFactory"


@dataclass
class ImageNavHabitatBaselinesRLConfig(HabitatBaselinesRLConfig):
    rl: ImageNavRLConfig = ImageNavRLConfig()
    vector_env_factory: ImageNavVectorEnvFactoryConfig = ImageNavVectorEnvFactoryConfig()

# -----------------------------------------------------------------------------
# Register configs in the Hydra ConfigStore
# -----------------------------------------------------------------------------
# Action
cs.store(
    package="habitat.task.actions.navigation_mode",
    group="habitat/task/actions",
    name="navigation_mode",
    node=NavigationModeActionConfig,
)
cs.store(
    package="habitat.task.actions.arm_action",
    group="habitat/task/actions",
    name="arm_action",
    node=FindingDoryArmActionConfig,
)
cs.store(
    package="habitat.task.actions.pddl_intermediate_stop",
    group="habitat/task/actions",
    name="pddl_intermediate_stop",
    node=PDDlIntermediateStopActionConfig,
)

# Sensors
cs.store(
    package="habitat.task.lab_sensors.other_object_segmentation_sensor",
    group="habitat/task/lab_sensors",
    name="other_object_segmentation_sensor",
    node=OtherObjectSegmentationSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.time_of_day_sensor",
    group="habitat/task/lab_sensors",
    name="time_of_day_sensor",
    node=TimeOfDaySensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.lang_memory_goal_sensor",
    group="habitat/task/lab_sensors",
    name="lang_memory_goal_sensor",
    node=LangMemoryGoalSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.last_action_sensor",
    group="habitat/task/lab_sensors",
    name="last_action_sensor",
    node=LastActionSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.can_take_action_sensor",
    group="habitat/task/lab_sensors",
    name="can_take_action_sensor",
    node=CanTakeActionSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.imagegoal_rotation_sensor",
    group="habitat/task/lab_sensors",
    name="imagegoal_rotation_sensor",
    node=ImageGoalRotationSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.manipulation_mode",
    group="habitat/task/lab_sensors",
    name="manipulation_mode",
    node=ManipulationModeSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.agent_state",
    group="habitat/task/lab_sensors",
    name="agent_state",
    node=AgentStateSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.oracle_keyframes",
    group="habitat/task/lab_sensors",
    name="oracle_keyframes",
    node=OracleKeyframesSensorConfig,
)

# Measures
cs.store(
    package="habitat.task.measurements.predicate_task_success",
    group="habitat/task/measurements",
    name="predicate_task_success",
    node=PredicateTaskSuccessConfig,
)
cs.store(
    package="habitat.task.measurements.simple_reward",
    group="habitat/task/measurements",
    name="simple_reward",
    node=SimpleRewardConfig,
)
cs.store(
    package="habitat.task.measurements.angle_to_goal",
    group="habitat/task/measurements",
    name="angle_to_goal",
    node=AngleToGoalConfig,
)
cs.store(
    package="habitat.task.measurements.angle_success",
    group="habitat/task/measurements",
    name="angle_success",
    node=AngleSuccessConfig,
)
cs.store(
    package="habitat.task.measurements.train_success",
    group="habitat/task/measurements",
    name="train_success",
    node=TrainSuccessConfig,
)
cs.store(
    package="habitat.task.measurements.picked_obj_anywhere_on_goal",
    group="habitat/task/measurements",
    name="picked_obj_anywhere_on_goal",
    node=PickedObjAnywhereOnGoal,
)
cs.store(
    package="habitat.task.measurements.high_level_goal_success",
    group="habitat/task/measurements",
    name="high_level_goal_success",
    node=HighLevelGoalSuccessConfig,
)
cs.store(
    package="habitat.task.measurements.vlm_failure_modes",
    group="habitat/task/measurements",
    name="vlm_failure_modes",
    node=VLMFailureModeSuccessConfig,
)
cs.store(
    package="habitat.task.measurements.high_level_dtg_success",
    group="habitat/task/measurements",
    name="high_level_dtg_success",
    node=HighLevelDTGSuccessConfig,
)
cs.store(
    package="habitat.task.measurements.high_level_sem_cov_success",
    group="habitat/task/measurements",
    name="high_level_sem_cov_success",
    node=HighLevelSemCovSuccessConfig,
)
cs.store(
    package="habitat.task.measurements.subgoal_count",
    group="habitat/task/measurements",
    name="subgoal_count",
    node=SubGoalCountConfig,
)
cs.store(
    package="habitat.task.measurements.low_level_policy_failure_modes",
    group="habitat/task/measurements",
    name="low_level_policy_failure_modes",
    node=LowLevelPolicyFailureModeConfig,
)
cs.store(
    package="habitat.task.measurements.findingdory_spl",
    group="habitat/task/measurements",
    name="findingdory_spl",
    node=FindingDorySPLMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.oracle_solution_generator",
    group="habitat/task/measurements",
    name="oracle_solution_generator",
    node=OracleSolutionGeneratorConfig,
)
cs.store(
    package="habitat.task.measurements.findingdory_high_level_spl",
    group="habitat/task/measurements",
    name="findingdory_high_level_spl",
    node=FindingDoryHighLevelSPLConfig,
)

# Task
cs.store(
    package="habitat.task",
    group="habitat/task",
    name="findingdory_task_config_base",
    node=FindingDoryTaskConfig,
)
cs.store(
    package="habitat.task",
    group="habitat/task",
    name="image_nav_task_config_base",
    node=ImageNavTaskConfig,
)

# Dataset
cs.store(
    package="habitat.dataset",
    group="habitat/dataset",
    name="findingdory_dataset_config_schema",
    node=FindingDoryDatasetConfig,
)

# Policy
cs.store(
    package="habitat_baselines.rl.policy",
    name="imagenav_policy_base",
    node=ImageNavPolicyConfig,
)
cs.store(
    group="habitat_baselines",
    name="imagenav_habitat_baselines_config_base",
    node=ImageNavHabitatBaselinesRLConfig(),
)

class FindingDoryConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="findingdory",
            path="pkg://findingdory/config/",
        )

register_hydra_plugin(FindingDoryConfigPlugin)
