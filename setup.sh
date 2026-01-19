# Run from root directory of findingdory project folder
conda_env_name=findingdory

# Create conda env and install habitat-sim
mamba create -n $conda_env_name python=3.9 cmake=3.14.0 -y
mamba install -n $conda_env_name habitat-sim=0.3.1 withbullet headless -c conda-forge -c aihabitat -y

mamba activate $conda_env_name

git submodule update --init --recursive
pip install -e third_party/habitat-lab/habitat-lab
pip install -e third_party/habitat-lab/habitat-baselines

# Install vc_models
pip install git+https://github.com/facebookresearch/eai-vc.git@main#subdirectory=vc_models

# Install the repo as a package
pip install -e .

# install dependencies for evaluating VLMs
pip install -e .[vlm_baseline]

# install pytorch3d, can be replaced by mamba if conda is slow
conda install pytorch3d

# install dependencies for the mapping baseline
pip install -e .[mapping_baseline]
