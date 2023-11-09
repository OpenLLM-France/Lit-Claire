---
language:
- fr
license: apache-2.0
pipeline_tag: text-generation
tags:
- pretrained
inference:
    parameters:
        temperature: 1.0
        max_new_tokens: 200
        top_k: 10
---

# Claire-7B-Apache-0.1

**Claire-7B-Apache-0.1 is a 7B parameter causal decoder-only model built by [OpenLLM-France](https://github.com/OpenLLM-France)**
**adapted from [Falcon-7b](https://huggingface.co/tiiuae/falcon-7b) on French conversational data.**

**It is made available under the Apache 2.0 license.**

## How to use

```python
import transformers
import torch

model = "OpenLLM-France/Claire-7B-Apache-0.1"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True                          # For efficient inference, if supported by the GPU card
)

pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
generation_kwargs = dict(
    num_return_sequences=1,                    # Number of variations to generate.
    return_full_text= False,                   # Do not include the prompt in the generated text.
    max_new_tokens=200,                        # Maximum length for the output text.
    do_sample=True, top_k=10, temperature=1.0, # Sampling parameters.
    pad_token_id=tokenizer.eos_token_id,       # Just to avoid a harmless warning.
)

prompt = """\
- Bonjour Dominique, qu'allez-vous nous cuisiner aujourd'hui ?
- Bonjour Camille,\
"""
completions = pipeline(prompt, **generation_kwargs)
for completion in completions:
    print(prompt + " […]" + completion['generated_text'])
```
This will print something like:
```
- Bonjour Dominique, qu'allez-vous nous cuisiner aujourd'hui ?
- Bonjour Camille, […] je vous prépare un plat de saison, une daube provençale.
- Ah je ne connais pas cette recette.
- C'est très facile à préparer, vous n'avez qu'à mettre de l'eau dans une marmite, y mettre de l'oignon émincé, des carottes coupées en petits morceaux, et vous allez mettre votre viande de bœuf coupé en petits morceaux également.
- Je n'ai jamais cuisiné de viande de bœuf, mais c'est vrai que ça a l'air bien facile.
- Vous n'avez plus qu'à laisser mijoter, et ensuite il sera temps de servir les clients.
- Très bien.
```

You will need at least 5GB of VRAM to run inference using 4bit quantization (16GB of VRAM without 4bit quantization).

If you have troubles running this code, make sure you have recent versions of `torch`, `transformers` and `accelerate` (see [requirements.txt](requirements.txt)).

## Training Details

### Training Data

Claire-7B-Apache-0.1 was tuned from Falcon-7b on the following data distribution:

| **Data source**                         | **Words**  | **Training Sampling Weight** | **Sources**                               |
|-----------------------------------------|------------|------------------------------|-------------------------------------------|
| Assemblée Nationale                     | 135M       | 57%                          | assemblee-nationale.fr                    |
| Theatre                                 |  16M       | 28.5%                        | theatre-classique.fr, theatregratuit.com  |
| Meetings                                |   1.0M     | 10.5%                        | SUMM-RE, LinTO                            |
| Debates                                 |   326k     |  3.4%                        | FreD                                      |
| Presentation, Conversations             |    58k     | <1%                          | LinTO                                     |

The data was tokenized with the [Falcon-7b](https://huggingface.co/tiiuae/falcon-7b) tokenizer.

The model has been trained and evaluated on French dialogues but may be able to generate conversations in other languages from the original Falcon-7B training data.

### Training Procedure 

Claire-7B-Apache-0.1 is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).
See [Falcon-7b](https://huggingface.co/tiiuae/falcon-7b) for more details.

Claire-7B-0.1 was trained on A100 80GB during about 50 GPU hours.

Hyperparameters were the following:

| **Hyperparameter** | **Value**  |
|--------------------|------------|
| Precision          | `bfloat16` |
| Optimizer          | AdamW      |
| Learning rate      | 1e-4       |
| Weight decay       | 1e-2       |
| Batch size         | 132        |
| LoRA rank          | 16         |
| LoRA alpha         | 32         |
| Dropout            | 0.05       |
| gradient clipping  | 1          |

## License

Claire-7B-Apache-0.1 is made available under the Apache 2.0 license.

## Acknowledgements

This work was performed using HPC resources from GENCI–IDRIS (Grant 2023-AD011014561). 

## Contact

contact@openllm-france.fr