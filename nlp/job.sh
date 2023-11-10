#!/bin/bash
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-08:00
#SBATCH --mail-user=<stephen.lu@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=1

cd $project/degenerate-attn/nlp
module purge
module load python/3.10 scipy-stack
source ~/pyenv/degeneracy/bin/activate

source .env
group=results/wikitext2/vanilla-transformer

mkdir -p group

# Train the models
# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --degenerate --save $group/degenerate-seed-43.pt --seed 43
python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --degenerate --save $group/degenerate-seed-7.pt --seed 7
# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --degenerate --save $group/degenerate-seed-113.pt --seed 113

# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --save $group/normal-seed-43.pt --seed 43
# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --save $group/normal-seed-7.pt --seed 7
# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --save $group/normal-seed-113.pt --seed 113

python main.py --cuda --bptt 512 --emsize 600 --nhid 600 --nhead 6 --nlayers 6 --epochs 200 --model Transformer --lr 5 --wandb --degenerate --save $group/degenerate-large-seed-55.pt --seed 55
python main.py --cuda --bptt 512 --emsize 600 --nhid 600 --nhead 6 --nlayers 6 --epochs 200 --model Transformer --lr 5 --wandb --save $group/normal-large-seed-55.pt --seed 55

# Generate some sample text with each model
python generate.py --cuda --checkpoint $group/degenerate-seed-43.pt --outf $group/degenerate-seed-43.txt --words 1000 --temperature 1 --seed 43
python generate.py --cuda --checkpoint $group/normal-seed-43.pt --outf $group/normal-seed-43.txt --words 1000 --temperature 0.8 --seed 43
