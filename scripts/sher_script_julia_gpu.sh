#!/bin/bash
#
#SBATCH --job-name=nmrnn
#SBATCH --partition=normal,hns,owners,swl1
#SBATCH --time=4:0:0
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G

# activate virtual environment
ml python/3.9
source /home/groups/swl1/nm-rnn/.venv/bin/activate

# run the sweep!
#wandb agent nm-rnn/nm-rnn-mwg/a761y26q
#wandb agent nm-rnn/nm-rnn-mwg/z14v7b8t
#wandb agent nm-rnn/nm-rnn-mwg/ybrfgslo
#wandb agent nm-rnn/nm-rnn-mwg/6q35efsh
wandb agent nm-rnn/nm-rnn-mwg/nmpc45l4
