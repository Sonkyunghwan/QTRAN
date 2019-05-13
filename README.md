# QTRAN

There will be additional updates later

## Predator-prey

Training

$algorithm = vdn, qmix, pqmix5(=QTRAN-alt in the paper), pqmix7(=QTRAN in the paper)

(i) 2 Predator & 1 Prey (5X5 Map) with P=0.5

python main.py --scenario endless3 --n_predator 2 --n_prey1 0 --n_prey2 1 --n_prey 1 --map_size 5 --agent pos_cac_fo --training_step 3000000 --testing_step 10000 --max_step 100 --b_size 600000 --df 0.99 --eval_step 100 --algorithm $algorithm --lr 0.0005 --seed 0 --penalty 5 --comment 215

(ii) 4 Predator & 2 Prey (7X7 Map) with P=0.5

python main.py --scenario endless3 --n_predator 4 --n_prey1 0 --n_prey2 2 --n_prey 2 --map_size 7 --agent pos_cac_fo --training_step 6000000 --testing_step 10000 --max_step 100 --b_size 1000000 --df 0.99 --eval_step 100 --algorithm $algorithm --lr 0.0005 --seed 0 --penalty 5 --comment 427 &


## Others

Training

$algorithm = vdn, qmix, pqmix5(=QTRAN-alt in the paper), pqmix7(=QTRAN in the paper)

python main.py --agent pos_cac_fo --training_step 10000 --b_size 10000 --m_size 32 --seed 0 --algorithm $algorithm --penalty 0


In make_env.py

(i) Matrix game

from envs.environment import MultiAgentSimpleEnv2 as MAS

(i) Gaussian Squeeze

from envs.environment import MultiAgentSimpleEnv4 as MAS
