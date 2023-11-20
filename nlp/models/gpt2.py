from transformers import GPT2LMHeadModel, AutoConfig, GPT2Config

def get_raw_model(vocab, args):

    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=args.bptt,
        n_embd=args.emsize,
        n_layer=args.nlayers,
        n_head=args.nhead,
        n_inner=args.nhid,
        activation_function='gelu_new',
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=0,
        # attn_pdrop=args.dropout,
        scale_attn_weights=True,
        degenerate_attn=False,
        bos_token_id=vocab.word2idx['<bos>'] if '<bos>' in vocab.word2idx else None,
        eos_token_id=vocab.word2idx['<eos>'] if '<eos>' in vocab.word2idx else None,
    )

    model = GPT2LMHeadModel(config)
    model.model_type = 'GPT2'

    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    return model


def get_pretrained_model():
    model = GPT2LMHeadModel.from_pretrained("./gpt2")
    return model
