import torch, transformers
from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM


class EndpointHandler:
    def __init__(self, path=""):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto", torch_dtype=torch.bfloat16, load_in_4bit=True
        )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, str]:
        # process input
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        # overwrite parameters
        parameters = {
            "max_length": 128,
            "do_sample": True,
            "top_k": 10,
            "temperature": 1.0,
            "return_full_text": False
        }

        sequences = self.pipeline(
            inputs,
            **parameters
        )

        return [{"generated_text": sequences[0]["generated_text"]}]
