#!/bin/bash

source /home/user/conda/bin/activate outpace
cd /home/user/outpace_diffusion/

sudo chown -R user:user /home/user/outpace_diffusion/

python outpace_train.py env=${env_name} num_train_steps=${num_train_steps} seed=${seed_number}

