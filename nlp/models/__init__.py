from .vanilla_rnn import *
from .vanilla_transformer import *
from .gpt2 import *

models = {
    'rnn': RNNModel,
    'transformer': TransformerModel,
    'gpt2': {
        'raw': get_raw_model,
        'pretrained': get_pretrained_model,
    },
}
