#!/usr/bin/env python3

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(
        description='Test pretrained model in litgit format, by giving likelihood of sentences and performing completion on it',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('text', type=str, help='sentence to complete', nargs="*")
    parser.add_argument('--model', type=str, help='model name',
        #default="/home/jlouradour/projects/OpenLLM/checkpoints/EleutherAI/pythia-70m",
        default="/gpfswork/rech/qgz/commun/Claire/checkpoints/tiiuae/falcon-7b",
    )
    parser.add_argument('--length_generation', type=int, default=50, help='Number of tokens to generate')
    parser.add_argument('--bos', default=False, action="store_true", help='Include BOS token')
    parser.add_argument('--eos', default=False, action="store_true", help='Include EOS token')
    args = parser.parse_args()

# Ugly path
import sys, os
wd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path += [wd + "/lit_gpt", wd]

import torch
import lightning as L
import lit_gpt

def load_model(
    checkpoint_dir,
    strategy: str = "auto",
    devices: int = 1,
    precision: str = None,
    quantize: str = None,
    verbose: bool = False,
    ):
    from lit_gpt import GPT, Config
    from lit_gpt.tokenizer import Tokenizer
    from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, load_checkpoint, quantization
    from pathlib import Path
    import time

    # See https://github.com/Lightning-AI/lit-gpt/blob/main/generate/base.py

    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)

    check_valid_checkpoint_dir(checkpoint_dir)

    precision = precision or get_default_supported_precision(training=False)

    if strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False)
    fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
    fabric.launch()
    
    config = Config.from_json(checkpoint_dir / "lit_config.json")

    if quantize is not None and devices > 1:
        raise NotImplementedError
    if quantize == "gptq.int4":
        model_file = "lit_model_gptq.4bit.pth"
        if not (checkpoint_dir / model_file).is_file():
            raise ValueError("Please run `python quantize/gptq.py` first")
    else:
        model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    if verbose: fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True), quantization(quantize):
        model = GPT(config)
    if verbose: fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    t0 = time.perf_counter()

    if quantize:
        # for quantization, need to load before moving to device
        load_checkpoint(fabric, model, checkpoint_path, strict=(quantize is None))

    model.eval()
    model = fabric.setup_module(model)

    if not quantize:
        load_checkpoint(fabric, model, checkpoint_path, strict=(quantize is None))

    if verbose: fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    tokenizer = Tokenizer(Path(model_folder))

    torch.set_float32_matmul_precision("high")
    L.seed_everything(1234)

    return fabric, tokenizer, model

@torch.inference_mode()
def generate(
    model,
    idx,
    max_returned_tokens,
    temperature = 1.0,
    top_k = None,
    eos_id = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    T = idx.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    # generate up to a fixed number of tokens
    for _ in range(max_returned_tokens - T):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx
    
def init_model(fabric, model, max_returned_tokens):
    with fabric.init_tensor():
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens

if __name__ == "__main__":

    args_texts = args.text       
    model_folder = args.model
    bos = args.bos
    eos = args.eos

    texts = []
    previous_is_file = True
    for t in args_texts:
        if os.path.isfile(t):
            with open(t, "r") as f:
                texts.append(f.read())
            previous_is_file = True
        elif not previous_is_file:
            texts[-1] += " " + t
        else:
            texts.append(t)
    if not len(texts):
        # Default texts to test with
        texts = [
            "Bonjour, qu'en dis-tu si",
            "sqdfslpgqaf sdxxbfghzdf reeeepgb pplmsdhhjnh",
            "Il ne faut surtout pas croire que",
            "La vie est belle, mais",
            "Je suis un peu fatigué, je vais",
            "Je ne sais pas quoi faire, je ne peux",
            "J'en",
            "Freddy Mercury est le chanteur du groupe de rock Queen.",
            "Il y a beaucoup de rumeurs autour de la mort de Jimi Hendrix.",
            "Dominique de Villepin a été nommé Premier ministre en remplacement de Jean-Pierre Raffarin le 31 mai 2005.",
            "Edith Cression fut la première femme nommée au poste de Premier ministre en France.",
        ]
    
    # Load model    
    fabric, tokenizer, model = load_model(model_folder)

    # Get the maximum length
    tokens_list = [tokenizer.encode(text, bos=bos, eos=eos) for text in texts]
    max_returned_tokens = args.length_generation + max([len(tokens) for tokens in tokens_list])

    for text in texts:

        tokens = tokenizer.encode(text, bos=bos, eos=eos, device=fabric.device)

        init_model(fabric, model, max_returned_tokens)

        logits = model(tokens.reshape(1,-1))[0]
        logprobs = logits.log_softmax(-1)
        i_start = 1 if bos else 0
        # Take probabilities at given indices
        logprobs = logprobs[torch.arange(i_start,len(tokens)-1), tokens[i_start+1:]]
        # Average over tokens
        avg_logprob = logprobs.mean(-1).item()

        print("=====================================")
        print(f"({avg_logprob:6.3f}) {text}")
        
        # Predict the next tokens
        if not args.length_generation:
            continue

        if eos:
            tokens = tokens[:-1]
            logits = logits[:-1]

        tokens = generate(model, tokens, args.length_generation + len(tokens), temperature=1, top_k=1)
            
        # last_logit = logits[-1]
        # for i in range(args.length_generation):
        #     next_token = last_logit.argmax()
        #     tokens = torch.cat([tokens, next_token.reshape(1)])
        #     if i == args.length_generation - 1:
        #         break
        #     # Very naive and unefficient implementation...
        #     last_logit = model(tokens.reshape(1,-1))[0, -1]

        generated = tokenizer.decode(tokens)[len(text):]
        
        print("...", generated)

