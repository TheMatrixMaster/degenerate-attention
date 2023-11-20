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
module load gcc arrow cuda scipy-stack python/3.10
source ~/pyenv/degeneracy/bin/activate

source .env
group=results/wikitext2/gpt2

mkdir -p $group

# Train the models
# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --degenerate --save $group/degenerate-seed-43.pt --seed 43
# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --degenerate --save $group/degenerate-seed-7.pt --seed 7
# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --degenerate --save $group/degenerate-seed-113.pt --seed 113

# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --save $group/normal-seed-43.pt --seed 43
# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --save $group/normal-seed-7.pt --seed 7
# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --save $group/normal-seed-113.pt --seed 113

# python main.py --cuda --bptt 512 --emsize 600 --nhid 600 --nhead 6 --nlayers 6 --epochs 50 --model GPT2 --lr 5 --wandb --degenerate --save $group/degenerate-seed-55.pt --seed 55
# ython main.py --cuda --bptt 512 --emsize 600 --nhid 600 --nhead 6 --nlayers 6 --epochs 50 --model GPT2 --lr 5 --wandb --degenerate --save $group/degenerate-seed-32.pt --seed 32
python main.py --cuda --bptt 512 --emsize 600 --nhid 600 --nhead 6 --nlayers 6 --epochs 100 --model GPT2 --lr 5 --seed 55
python main.py --cuda --bptt 512 --emsize 600 --nhid 600 --nhead 6 --nlayers 6 --epochs 100 --model GPT2 --lr 5 --seed 32

# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --degenerate --save $group/degenerate-seed-43.pt --seed 43 --data './data/wikitext-103'
# python main.py --cuda --epochs 50 --model Transformer --lr 5 --wandb --save $group/normal-seed-43.pt --seed 43 --data './data/wikitext-103'

# Generate some sample text with each model
# python generate.py --cuda --checkpoint $group/degenerate-seed-55.pt --outf $group/degenerate-seed-55.txt --words 1000 --temperature 0.8 --seed 55
# python generate.py --cuda --checkpoint $group/normal-seed-55.pt --outf $group/normal-seed-55.txt --words 1000 --temperature 0.8 --seed 55
