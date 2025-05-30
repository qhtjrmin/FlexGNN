#!/bin/bash

# Docker setting
IMAGE_NAME=flexgnn
GPU_DEVICES="device=0,1"

# Dataset and config paths
DATA_DIR=$(pwd)/dataset
CONFIG_DIR=$(pwd)/configs

# Experiment settings
DATASET=products
INPUT_PATH=/dataset/${DATASET}/
CONFIG_PATH=/configs/${DATASET}/2hop_256feat.ini
EPOCHS=10
MODEL=GCN

# Run docker container
docker run --rm --gpus "\"$GPU_DEVICES\"" \
  -v ${DATA_DIR}:/dataset \
  -v ${CONFIG_DIR}:/configs \
  ${IMAGE_NAME} \
  ./flexgnn \
  -input_path ${INPUT_PATH} \
  -config_path ${CONFIG_PATH} \
  -e ${EPOCHS} \
  -model ${MODEL}

