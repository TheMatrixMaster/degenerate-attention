# degenerate-attention

Instead of performing softmax over the last dimension of the attention scores which corresponds to the keys dimension, we softmax over the penultimate dimension of the attention scores which corresponds to the queries dimension. This is equivalent to performing softmax over the keys dimension, which forces the sum of attention scores in each column to equal 1. In other words, this forces the queries to compete for the keys instead of the keys competing for the queries, which may yield better results according to the global workspace theory.

## Usage

### Setup
You can install our flavour of torch with degenerate attention by either installing from source, or by making a local installation, then porting over the degenerate attention code files. For the second option to work, we need to enforce a consistent version of torch, so we use version 2.0.1.
```bash
# from source
. setup_from_src.sh

# local installation
. setup_from_cpy.sh
```

### Training
We will setup multiple tasks on which to train different transformers with degenerate attention. The first task consists of language modelling with a variety of natural language datasets. 

#### Language Modelling
To run a language model, there are a variety of hyper parameters that you can set. Please refer to the [original README](nlp/README.md) for more information. Here is an example where we run a vanilla transformer on the wikitext2 dataset.

```bash
cd nlp
python main.py --cuda --epochs 6 --model Transformer --lr 5 --wandb --degenerate
python main.py --cuda --epochs 6 --model Transformer --lr 5 --wandb
```