#!/bin/bash
#
#SBATCH --job-name=nmrnn
#SBATCH --partition=normal,hns,owners,swl1
#SBATCH --time=4:0:0
#SBATCH --ntasks=1
#SBATCH --mincpus=32
#SBATCH --mem-per-cpu=1G

# activate virtual environment
ml python/3.9
source /home/groups/swl1/nm-rnn/.venv/bin/activate

# run the sweep!
#wandb agent nm-rnn/nm-rnn-mwg/a761y26q
#wandb agent nm-rnn/nm-rnn-mwg/z14v7b8t
#wandb agent nm-rnn/nm-rnn-mwg/ybrfgslo
#wandb agent nm-rnn/nm-rnn-mwg/9j4aioc6
#wandb agent nm-rnn/nm-rnn-mwg/b4wl20nr
#python3 train_multitask.py
#wandb agent nm-rnn/nm-rnn-multitask/dpivmlbz
#wandb agent nm-rnn/nm-rnn-multitask/14nzxzaj
#wandb agent nm-rnn/nm-rnn-multitask/m2cv1qxy
#wandb agent nm-rnn/vanilla-rnn-mwg/j89w2gxe
#wandb agent nm-rnn/multitask-lstm/q8u19ysl
#wandb agent nm-rnn/multitask-lstm/5ewltj7a
wandb agent nm-rnn/lstm-mwg/fj9jlfvf
