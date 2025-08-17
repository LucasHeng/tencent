#!/bin/bash

export TRAIN_LOG_PATH="/home/lucasheng/Tencent/logs/"
export TRAIN_TF_EVENTS_PATH="/home/lucasheng/Tencent/tf_events/"
export TRAIN_DATA_PATH="/home/lucasheng/Tencent/data/TencentGR_1k/"
export TRAIN_CKPT_PATH="/home/lucasheng/Tencent/ckpt/"

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
python -u main.py --hidden_units 128 --num_blocks 4 --num_heads 4 --l2_emb 0.1 --num_epochs 4 --use_hstu_attn --sample_neg_num 2 --use_all_in_batch