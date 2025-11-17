#!/bin/bash
all_ep_ids=($(seq 1 100))
chunk_size=768

# Get all episode IDs as comma-separated string
all_ep_ids_str=""
for j in $(seq 0 $((${#all_ep_ids[@]} - 1))); do
    all_ep_ids_str="${all_ep_ids_str}${all_ep_ids[$j]},"
done
all_ep_ids_str=${all_ep_ids_str%,} # Remove trailing comma

# Create base output folder
base_folder="findingdory/data/findingdory_outputs/qwen_imagenav/slurm_logs"
folder_name="${base_folder}/batch_0"
mkdir -p "$folder_name"

export HABITAT_SIM_LOG=quiet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python findingdory/run_findingdory_eval.py \
    --config-name=baseline/qwen_imagenav.yaml \
    habitat.dataset.episode_ids=[${all_ep_ids_str}] \
    habitat.task.selected_instruction=0 \
    habitat_baselines.agent.config.vlm_temp_folder="data/findingdory_outputs/qwen_imagenav/temp" \
    habitat_baselines.agent.config.output_folder="data/findingdory_outputs/qwen_imagenav/logs" \
    habitat_baselines.agent.config.chunk_size=${chunk_size} \
    habitat_baselines.agent.config.subsample_frames=True