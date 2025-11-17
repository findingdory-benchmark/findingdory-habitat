#!/usr/bin/env bash
# Adapted from https://github.com/facebookresearch/home-robot/blob/main/download_data.sh

mkdir -p data/datasets

cd data
git clone https://huggingface.co/datasets/yali30/hssd-hab --recursive

git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects objects --recursive
cd objects
git checkout e9b714c37ebcc55f4702a8199fd92d6b5432553a

cd ../datasets
git clone https://huggingface.co/datasets/yali30/findingdory-habitat --recursive
cd ../

cp datasets/findingdory-habitat/findingdory/fpModels-v0.2.3.csv objects/

echo ""
echo "Download the robot model Habitat uses..."
mkdir -p robots/hab_stretch
cd robots/hab_stretch
wget --no-check-certificate http://dl.fbaipublicfiles.com/habitat/robots/hab_stretch_v1.0.zip
unzip -o hab_stretch_v1.0.zip
