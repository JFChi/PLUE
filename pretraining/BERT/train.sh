#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
CURRENT_DIR=$(pwd)
ROOT_DIR=$(realpath ../..)
DATA_DIR=${ROOT_DIR}/pretraining/data

OUT_DIR=$CURRENT_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUs=8

python -m torch.distributed.launch --nproc_per_node $NUM_GPUs --use_env run_mlm_no_trainer.py \
    --model_name_or_path bert-base-uncased \
    --train_file $DATA_DIR/train.txt \
    --validation_file $DATA_DIR/test.txt \
    --output_dir $OUT_DIR \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --max_train_steps 100000 \
    --num_warmup_steps 1000 \
    --save_interval_updates 5000 \
    2>&1 | tee $OUT_DIR/training.log
