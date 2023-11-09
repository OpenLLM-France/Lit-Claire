---
language:
- fr
license: cc-by-nc-sa-4.0
pipeline_tag: text-generation
tags:
- pretrained
inference:
    parameters:
        temperature: 1.0
        max_new_tokens: 200
        top_k: 10
---

# Claire-Mistral-7B-0.1

**Claire-Mistral-7B-0.1 is a 7B parameter causal decoder-only model built by [OpenLLM-France](https://github.com/OpenLLM-France)**
**adapted from [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) on French conversational open data.**

**Given that some of the corpora used for training are only available under CC-BY-NC-SA licenses, Claire-7B-0.1 is also made available under a CC-BY-NC-SA license.**


## How to use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "OpenLLM-France/Claire-Mistral-7B-0.1"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   "[Intervenant 1:] Bonjour, pouvez-vous nous parler de votre sport préféré ?\n[Intervenant 2:] Alors euh oui,",
    max_new_tokens=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

```

For fast inference with Claire, check-out [Text Generation Inference](https://github.com/huggingface/text-generation-inference).

You will need **at least 16GB of memory** to swiftly run inference with Claire-Mistral-7B-0.1.

## Training Details

### Training Data

Claire-Mistral-7B-0.1 was tuned from Mistral-7B-v0.1 on the following data distribution:

| **Data source**               | **Words**  | **Training Sampling Weight** | **Sources**                                         |
|-------------------------------|------------|------------------------------|-----------------------------------------------------|
| Assemblée Nationale           | 135M       | 35%                          | assemblee-nationale.fr                              |
| Theatre                       |  16M       | 18%                          | theatre-classique.fr, theatregratuit.com            |
| Interviews                    |   6.4M     | 29%                          | TCOF, CFPP, CFPB, ACSYNT, PFC, Valibel (ORFEO), ESLO              |
| Free Conversations            |   2.2M     | 10%                          | CRFP, OFROM, CID, Rhapsodie, ParisStories, PFC, CLAPI, C-ORAL-ROM (ORFEO), LinTO, ESLO |
| Meetings                      |   1.2M     |  5%                          | SUMM-RE, LinTO, ORFEO réunions de travail |
| Debates                       |   402k     | <2%                          | FreD, ESLO                                |
| Assistance                    |   159k     | <1%                          | ORFEO fleuron, UBS, OTG, ESLO             |
| Presentation, Address         |    86k     | <0.5%                        | Valibel (ORFEO), LinTO, ESLO              |

The data was tokenized with the [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) tokenizer.

The model has been trained and evaluated on French dialogues but may be able to generate conversations in other languages from the original Falcon-7B training data.

### Training Procedure 

Claire-Mistral-7B-0.1 is a causal decoder-only model trained on a causal language modeling task (i.e., predict the next token).
See [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) for more details.

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

Claire-Mistral-7B-0.1 is made available under the CC-BY-NC-SA 4.0 license.

## Acknowledgements

This work was performed using HPC resources from GENCI–IDRIS (Grant 2023-AD011014561). 

## Contact

contact@openllm-france.fr