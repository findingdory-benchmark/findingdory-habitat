#!/bin/bash

all_ep_ids=($(seq 1 100))
N_PARTS=10
exp_idx=0
chunk_size=768

# Divide episodes into N parts and run on different GPUs
for i in $(seq 0 $(($N_PARTS - 1)))
do
    # Calculate index range for this batch
    total_eps=${#all_ep_ids[@]}
    start_idx=$((i * total_eps / $N_PARTS))
    end_idx=$(((i + 1) * total_eps / $N_PARTS - 1))
    
    # Get episode IDs for this batch
    batch_eps=""
    for j in $(seq $start_idx $end_idx); do
        batch_eps="${batch_eps}${all_ep_ids[$j]},"
    done
    batch_eps=${batch_eps%,} # Remove trailing comma

    # Create base output folder for this batch
    base_folder="findingdory/data/findingdory_outputs/qwen_imagenav/slurm_logs/batch_${exp_idx}_${i}"
    folder_name="${base_folder}/batch_${i}"
    mkdir -p "$folder_name"

    # Submit job for this batch
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=mmqwen_${i}
#SBATCH --output=$folder_name/val-%j.out
#SBATCH --error=$folder_name/val-%j.err
#SBATCH --gpus=a40:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --qos="short"
#SBATCH --requeue
#SBATCH --signal=USR1@100

export HABITAT_SIM_LOG=quiet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda activate findingdory

python findingdory/run_findingdory_eval.py \
    --config-name=baseline/qwen_imagenav.yaml \
    habitat.dataset.episode_ids=[${batch_eps}] \
    habitat.task.selected_instruction=0 \
    habitat_baselines.agent.config.vlm_temp_folder="data/findingdory_outputs/qwen_imagenav/temp/batch_${exp_idx}_${i}" \
    habitat_baselines.agent.config.output_folder="data/findingdory_outputs/qwen_imagenav/logs/batch_${exp_idx}_${i}" \
    habitat_baselines.agent.config.chunk_size=${chunk_size} \
    habitat_baselines.agent.config.subsample_frames=True
EOF

done