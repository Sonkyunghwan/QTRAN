#!/bin/sh

GPU=$1
#for penalty in 0 2 4 6 8 10 12 14
for seed in 28 29 30 31 32 #12 13
#for penalty in 5 10 15
do
for penalty in 5
do

#CUDA_VISIBLE_DEVICES=$1 python main.py --scenario endless3 --n_predator 2 --n_prey1 0 --n_prey2 1 --n_prey 1 --map_size 5 --agent $2 --training_step 3000000 --testing_step 10000 --max_step 100 --b_size 600000 --df 0.99 --eval_step 100 --algorithm $3 --lr 0.0005 --seed $seed --penalty $penalty --comment "$4"215 &






#for seed in 401 402 403 404 405
#do
    
#CUDA_VISIBLE_DEVICES=$1 python main.py --scenario endless3 --n_predator 3 --n_prey1 0 --n_prey2 2 --n_prey 2 --map_size 6 --agent $2 --training_step 3000000 --testing_step 10000 --max_step 100 --b_size 600000 --df 0.99 --eval_step 100 --algorithm $3 --lr 0.0005 --seed $seed --penalty $penalty --comment "$4"326 &

CUDA_VISIBLE_DEVICES=$1 python main.py --scenario endless3 --n_predator 4 --n_prey1 0 --n_prey2 2 --n_prey 2 --map_size 7 --agent $2 --training_step 6000000 --testing_step 10000 --max_step 100 --b_size 1000000 --df 0.99 --eval_step 100 --algorithm $3 --lr 0.0005 --seed $seed --penalty $penalty --comment "$4"427 &


#    CUDA_VISIBLE_DEVICES=$1 python main.py --scenario endless3 --n_predator 3 --n_prey 2 --map_size 7 --agent $2 --training_step 3000000 --testing_step 10000 --max_step 100 --b_size 500000 --df 0.99 --eval_step 100 --algorithm $3 --lr 0.0001 --seed $seed --penalty $penalty --beta $4 --comment "$4"-326 &

#CUDA_VISIBLE_DEVICES=$1 python main.py --scenario endless3 --n_predator 4 --n_prey 2 --map_size 8 --agent $2 --training_step 3000000 --testing_step 10000 --max_step 100 --b_size 500000 --df 0.99 --eval_step 100 --algorithm $3 --lr 0.0001 --seed $seed --penalty $penalty --comment "$4"428 &

#done

done
done
