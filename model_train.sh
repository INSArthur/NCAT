#!/usr/bin/env bash
python ./launch.py -data_path ./data/ -data_name Assist-2017 -environment env -CDM IRT1 -T 20 -ST [1,5,10,20] -agent Train -FA NCAT -latent_factor 128 \
-learning_rate 0.001 -training_epoch 300 -seed 145 -gpu_no 0 -inner_epoch 30 -rnn_layer 1 -gamma 0.8 -batch 128 -restore_model False

#data_name : assist1213