#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
python -u main.py --hidden_units 128 --num_blocks 6 --num_heads 8 --num_epochs 8 --use_hstu_attn --sample_neg_num 2 --norm_first --weight_decay 0.1