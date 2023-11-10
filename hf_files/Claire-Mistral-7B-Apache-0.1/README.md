---
language:
- fr
license: apache-2.0
pipeline_tag: text-generation
tags:
- pretrained
- conversational
widget:
  - text: |-
      - Bonjour Dominique, qu'allez-vous nous cuisiner aujourd'hui ?
      - Bonjour Camille,
    example_title: Request for a recipe
    group: Dash
  - text: |-
      [Intervenant 1:] Bonjour Dominique, qu'allez-vous nous cuisiner aujourd'hui ?
      [Intervenant 2:] Bonjour Camille,
    example_title: Request for a recipe
    group: Intervenant
  - text: |-
      [Camille:] Bonjour Dominique, qu'allez-vous nous cuisiner aujourd'hui ?
      [Dominique:] Bonjour Camille,
    example_title: Request for a recipe
    group: FirstName
  - text: |-
      [Camille Durand:] Bonjour Dominique, qu'allez-vous nous cuisiner aujourd'hui ?
      [Dominique Petit:] Bonjour Camille,
    example_title: Request for a recipe
    group: Named
inference:
    parameters:
        temperature: 1.0
        max_new_tokens: 200
        top_k: 10
---

# Claire-Mistral-7B-Apache-0.1

**Claire-Mistral-7B-Apache-0.1 is a 7B parameter causal decoder-only model built by [OpenLLM-France](https://github.com/OpenLLM-France)**
**adapted from [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) on French conversational data.**

It is made available under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

Note that this model is a variant of [Claire-Mistral-7B-0.1](https://huggingface.co/OpenLLM-France/Claire-Mistral-7B-0.1) which is trained on more French conversational data,
but published under the [CC-BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Typical usage

```python
import transformers
import torch

model_name = "OpenLLM-France/Claire-Mistral-7B-Apache-0.1"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True                          # For efficient inference, if supported by the GPU card
)

pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
generation_kwargs = dict(
    num_return_sequences=1,                    # Number of variants to generate.
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

You will need at least 6GB of VRAM to run inference using 4bit quantization (16GB of VRAM without 4bit quantization).

If you have trouble running this code, make sure you have recent versions of `torch`, `transformers` and `accelerate` (see [requirements.txt](requirements.txt)).

### Typical prompts

Claire-Mistral-7B-Apache-0.1 was trained on diarized French conversations. Please note that as a result, the model is prone to generate dialogues with disfluencies and other constructions common to spoken language.

During training, the dialogues were normalized in several formats. The possible formats for expected prompts are as follows:

A monologue can be specified as a single line prompt (though keep in mind that Claire might still return a dialogue because of its training):
```python
prompt = "Mesdames et messieurs les députés, chers collègues, bonsoir. Vous l'aurez peut-être remarqué, je cite rarement"
```

A dialogue between two speakers can be specified with one line per speech turn starting with a dash:
```python
prompt = """\
- Bonjour Dominique, qu'allez-vous nous cuisiner aujourd'hui ?
- Bonjour Camille,\
"""
```

A dialogue or multilogue (with two or more speakers) can be specified with lines that start with `[Intervenant X:]` where `X` is a number:
```python
prompt = """\
[Intervenant 1:] Bonjour Dominique, qu'allez-vous nous cuisiner aujourd'hui ?
[Intervenant 2:] Bonjour Camille,\
"""
```

A dialogue or multilogue with named speakers can be specified with lines that start with `[SpeakerName:]`
where `SpeakerName` can be a first name, a first and a last name, a nickname, a title…
```python
prompt = """\
[Mme Camille Durand:] Bonjour Dominique, qu'allez-vous nous cuisiner aujourd'hui ?
[Mr. Dominique Petit:] Bonjour Camille,\
"""
```

## Training Details

### Training Data

Claire-Mistral-7B-Apache-0.1 was tuned from Mistral-7B-v0.1 on the following data distribution:

| **Data type**                           | **Words**  | **Training Sampling Weight** | **Sources**                               |
|-----------------------------------------|------------|------------------------------|-------------------------------------------|
| Parliamentary Proceedings               | 135M       | 57%                          | assemblee-nationale.fr                    |
| Theatre                                 |  16M       | 28.5%                        | theatre-classique.fr, theatregratuit.com  |
| Meetings                                |   1.0M     | 10.5%                        | SUMM-RE, LinTO                            |
| Debates                                 |   326k     |  3.4%                        | FreD                                      |
| Presentation, Conversations             |    58k     | <1%                          | LinTO                                     |

The model has been trained and evaluated on French dialogues but may be able to generate conversations in other languages from the original Mistral-7B-v0.1 training data.

### Training Procedure 

Claire-Mistral-7B-Apache-0.1 is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).
See [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) for more details.

Claire-Mistral-7B-Apache-0.1 was trained on 8 A100 80GB for about 50 GPU hours.

Hyperparameters were the following:

| **Hyperparameter** | **Value**  |
|--------------------|------------|
| Precision          | `bfloat16` |
| Optimizer          | AdamW      |
| Learning rate      | 1e-4       |
| Weight decay       | 1e-2       |
| Batch size         | 128        |
| LoRA rank          | 16         |
| LoRA alpha         | 32         |
| Dropout            | 0.05       |
| gradient clipping  | 1          |

## Evaluation

See [Evaluation section of Claire-7B-0.1](https://huggingface.co/OpenLLM-France/Claire-7B-0.1#evaluation).

## License

Claire-Mistral-7B-Apache-0.1 is made available under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).

You can find a variant of this model trained on more data but published under the [CC-BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/)
at [OpenLLM-France/Claire-Mistral-7B-0.1](https://huggingface.co/OpenLLM-France/Claire-Mistral-7B-0.1).

## Acknowledgements

This work was performed using HPC resources from GENCI–IDRIS (Grant 2023-AD011014561). 

Claire-Mistral-7B-Apache-0.1 was created by members of [LINAGORA](https://labs.linagora.com/) (in alphabetical order): Ismaïl Harrando, Julie Hunter, Jean-Pierre Lorré, Jérôme Louradour, Michel-Marie Maudet, Virgile Rennard, Guokan Shang.

Special thanks to partners from the OpenLLM-France community, especially Christophe Cerisara (LORIA), Pierre-Carl Langlais and Anastasia Stasenko (OpSci), and Pierre Colombo, for valuable advice.

## Contact

contact@openllm-france.fr