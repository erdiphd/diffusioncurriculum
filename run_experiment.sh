#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointUMaze-v0 aim_disc_replay_buffer_capacity=10000 save_buffer=true adam_eps=0.01 seed=1 num_train_steps=40000 eval_frequency=1000
