#!/bin/sh

#for seed in 1 2 3 4 5
#do
    CUDA_VISIBLE_DEVICES=$1 python main.py --agent pos_cac_fo --training_step 8000 --b_size 10000 --m_size 64 --seed 6 --algorithm $2 --penalty $3 &
#done
    
