from transformers import GPT2LMHeadModel, AutoConfig


def get_raw_model(vocab_size: int, context_length: int):

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=vocab_size,
        n_ctx=context_length,
        # bos_token_id=,
        # eos_token_id=,
    )

    model = GPT2LMHeadModel(config)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    return model


def get_pretrained_model(checkpoint_path: str):
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    return model
