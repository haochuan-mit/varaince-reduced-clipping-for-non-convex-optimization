#!/bin/bash

# arguments
MODEL=resnet20 # options: fcn, resnet20, resnet56, resnet110, etc
DATASET=cifar10 # options: mnist, cifar10, cifar100
EPOCHS=400
RUNID=neural_nets # RUNID determines the name of directory of checkpoints
METRIC=top_1_acc # the learning rate is automatically tuned to get the best METRIC
DATANOISE=0.0 # variance of noise added to the image w.r.t. the L2 norm
LABELNOISE=0.0 # probability of replacing the label with a random one

# commands

# sgd
python run.py --run-id $RUNID --model $MODEL --method sgd_1024 --dataset $DATASET --lr 0.1 --tune-lr --epochs $EPOCHS -b 4096 --metric $METRIC --loss CE --deterministic --num-workers 0 --data_noise $DATANOISE --label_noise $LABELNOISE

# svrg
python run.py --run-id $RUNID --model $MODEL --method svrg_1024 --dataset $DATASET --lr 0.1 --tune-lr --epochs $EPOCHS -b 4096 --metric $METRIC --loss CE --deterministic --num-workers 0 --data_noise $DATANOISE --label_noise $LABELNOISE

# sarah
python run.py --run-id $RUNID --model $MODEL --method sarah_1024 --dataset $DATASET --lr 0.1 --tune-lr --epochs $EPOCHS -b 4096 --metric $METRIC --loss CE --deterministic --num-workers 0 --data_noise $DATANOISE --label_noise $LABELNOISE

# original spider (or spider1), where spider10d5_1024 means the clipping parameter is 0.5 
python run.py --run-id $RUNID --model $MODEL --method spider10d5_1024 --dataset $DATASET --lr 0.1 --tune-lr --epochs $EPOCHS -b 4096 --metric $METRIC --loss CE --deterministic --num-workers 0 --data_noise $DATANOISE --label_noise $LABELNOISE

# our (L0,L1)-spider (or spider2), where spider216d0=16_1024 means the clipping parameters are (0.5,16)
python run.py --run-id $RUNID --model $MODEL --method spider20d5=16_1024 --dataset $DATASET --lr 0.1 --tune-lr --epochs $EPOCHS -b 4096 --metric $METRIC --loss CE --deterministic --num-workers 0 --data_noise $DATANOISE --label_noise $LABELNOISE
