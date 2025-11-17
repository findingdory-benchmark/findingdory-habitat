#!/bin/bash
#SBATCH --job-name=mmbench
#SBATCH --output=slurm_logs/imagenav_dec_9/all_fixes_32_GPUs_frozen_enc_lr_1.5e-5/imagenav_2K-%j.out
#SBATCH --error=slurm_logs/imagenav_dec_9/all_fixes_32_GPUs_frozen_enc_lr_1.5e-5/imagenav_2K-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=a40:32
#SBATCH --cpus-per-task=12
#SBATCH --signal=USR1@300
#SBATCH --requeue

conda activate hab_memory_dev

export HABITAT_SIM_LOG=quiet

MAIN_ADDR=$(hostname -s)
export MAIN_ADDR

cd findingdory/findingdory

# create function to get EXP_DETAILS
get_exp_details () {
    local EXP_NAME=$1
    local WANDB_USER=$2
    EXP_FOLDER="runs/imagenav/${EXP_NAME}"
    mkdir -p ${EXP_FOLDER}
    export EXP_DETAILS="habitat_baselines.writer_type=wb \
    habitat_baselines.wb.project_name=findingdory \
    habitat_baselines.wb.entity=${WANDB_USER} \
    habitat_baselines.wb.run_name=${EXP_NAME} \
    habitat_baselines.tensorboard_dir=${EXP_FOLDER}/tb \
    habitat_baselines.video_dir=${EXP_FOLDER}/video_dir \
    habitat_baselines.eval_ckpt_path_dir=${EXP_FOLDER}/checkpoints \
    habitat_baselines.checkpoint_folder=${EXP_FOLDER}/checkpoints \
    habitat_baselines.log_file=${EXP_FOLDER}/train.log \
    habitat.dataset.data_path=data/datasets/findingdory_imagenav/findingdory/dataset/train/episodes.json.gz \
    habitat.dataset.viewpoints_matrix_path=data/datasets/findingdory_imagenav/findingdory/dataset/train/viewpoints.npy \
    habitat.dataset.transformations_matrix_path=data/datasets/findingdory_imagenav/findingdory/dataset/train/transformations.npy \
    habitat_baselines.num_environments=32 \
    habitat.simulator.rearrange_sim_close_threshold=1024 \
    habitat_baselines.rl.ppo.encoder_lr=1.5e-5 \
    habitat_baselines.rl.ddppo.pretrained_weights=data/datasets/findingdory_imagenav/pretrained_vis_enc/pretrained_ckpt.pth \
    habitat_baselines.rl.ddppo.pretrained_encoder=True \
    habitat_baselines.rl.ddppo.train_encoder=False \
    habitat.simulator.dataset_navmesh_dir=data/datasets/findingdory_imagenav/dataset/train/navmeshes_imagenav \
    habitat_baselines.load_resume_state_config=True"
}
set -x


EXP_NAME="imagenav_train"
WANDB_USER=""
get_exp_details $EXP_NAME $WANDB_USER
srun python -u -m run_imagenav --config-name=baseline/ddppo_imagenav.yaml \
    ${EXP_DETAILS}