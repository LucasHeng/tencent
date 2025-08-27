#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
python -u main.py --hidden_units 128 --num_blocks 6 --num_heads 8 --l2_emb 0.1 --num_epochs 4 --use_hstu_attn --sample_neg_num 2 --use_all_in_batch --norm_first