#!/bin/bash

# User-specified directory paths/checkpoint names to evaluate
EXP_NAME="imagenav_evals"
CKPT_DIR=""
CKPT_NAMES=("ckpt.0.pth" "ckpt.1.pth" "ckpt.2.pth" "ckpt.3.pth" "ckpt.4.pth" "ckpt.5.pth" "ckpt.6.pth" "ckpt.7.pth" "ckpt.8.pth" "ckpt.9.pth" "ckpt.10.pth")

# User-specified dataset paths/num_envs which will be used to override the checkpoint config during evals
DATASET_PATH="data/datasets/findingdory_imagenav/findingdory/dataset/val/episodes.json.gz"
VIEWPOINTS_FILE="data/datasets/findingdory_imagenav/findingdory/dataset/val/viewpoints.npy"
TRFS_PTH="data/datasets/findingdory_imagenav/findingdory/dataset/val/transformations.npy"
NUM_ENVS=12

# Iterate through each checkpoint and launch a separate Slurm job
for CKPT in "${CKPT_NAMES[@]}"; do
    CUR_CKPT_NAME=$(echo "${CKPT}" | sed 's/ckpt\.\([0-9]*\)\.pth/ckpt_\1/')
    FULL_EXP_NAME="${EXP_NAME}_${CUR_CKPT_NAME}"
    CKPT_PATH="${CKPT_DIR}/${CKPT}"
    EXP_FOLDER="runs/imagenav/${FULL_EXP_NAME}"
    mkdir -p "${EXP_FOLDER}"

    # Slurm script content
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=mmbench_${CUR_CKPT_NAME}
#SBATCH --output=imagenav_evals/val/${EXP_NAME}/${CUR_CKPT_NAME}/imagenav_val-%j.out
#SBATCH --error=imagenav_evals/val/${EXP_NAME}/${CUR_CKPT_NAME}/imagenav_val-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --signal=USR1@300
#SBATCH --requeue

conda activate hab_memory_dev

export HABITAT_SIM_LOG=quiet

MAIN_ADDR=\$(hostname -s)
export MAIN_ADDR

cd findingdory

set -x
srun python -u -m run --config-name=baseline/ddppo_imagenav.yaml \
    habitat_baselines.writer_type=tb \
    habitat_baselines.evaluate=True \
    habitat_baselines.eval.video_option=[] \
    habitat_baselines.wb.project_name=findingdory \
    habitat_baselines.tensorboard_dir=${EXP_FOLDER}/tb \
    habitat_baselines.video_dir=${EXP_FOLDER}/video_dir \
    habitat_baselines.eval_ckpt_path_dir=${CKPT_PATH} \
    habitat_baselines.checkpoint_folder=${EXP_FOLDER}/checkpoints \
    habitat_baselines.log_file=${EXP_FOLDER}/val.log \
    habitat.dataset.data_path=${DATASET_PATH} \
    habitat.dataset.viewpoints_matrix_path=${VIEWPOINTS_FILE} \
    habitat.dataset.transformations_matrix_path=${TRFS_PTH} \
    habitat_baselines.num_environments=${NUM_ENVS}
EOF

    # Log the submission of the batch job
    echo "Batch job for checkpoint ${CUR_CKPT_NAME} has been submitted."
done
