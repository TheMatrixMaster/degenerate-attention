#!/bin/bash
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-01:00
#SBATCH --mail-user=<stephen.lu@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=1

cd $project/degenerate-attn/nlp
module purge
module load python/3.10 scipy-stack
source ~/pyenv/degeneracy/bin/activate

python main.py --cuda --epochs 6 --model Transformer --lr 5 --wandb --degenerate