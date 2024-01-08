#!/bin/bash
#
#SBATCH --job-name=nmrnn
#SBATCH --partition=normal,hns
#SBATCH --time=4:0:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G

# activate virtual environment
source /home/groups/swl1/nm-rnn/.venv/bin/activate

# run the sweep!
wandb agent nm-rnn/nm-rnn-mwg/a761y26q