#!/usr/bin/env bash
# Adapted from https://github.com/facebookresearch/home-robot/blob/main/download_data.sh

mkdir -p findingdory/data/datasets

cd findingdory/data

export HF_HUB_DOWNLOAD_TIMEOUT=3600
export HF_HUB_ETAG_TIMEOUT=3600
hf download --repo-type dataset yali30/hssd-hab --local-dir hssd-hab

git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects objects --recursive
cd objects
git checkout e9b714c37ebcc55f4702a8199fd92d6b5432553a

cd ../../
cp dataset/object_attribute_annotation.csv data/objects/

cd data/datasets
git clone https://huggingface.co/datasets/yali30/findingdory-habitat --recursive
cd ../

cp datasets/findingdory-habitat/findingdory/fpModels-v0.2.3.csv objects/

echo ""
echo "Download the robot model Habitat uses..."
mkdir -p robots/hab_stretch
cd robots/hab_stretch
wget --no-check-certificate http://dl.fbaipublicfiles.com/habitat/robots/hab_stretch_v1.0.zip
unzip -o hab_stretch_v1.0.zip

cd ../../
cat <<EOF > default.physics_config.json
{
    "physics_simulator": "bullet",
    "timestep": 0.008,
    "gravity": [0, -9.8, 0],
    "friction_coefficient": 0.4,
    "restitution_coefficient": 0.1,
    "rigid object paths": ["objects"]
}
EOF