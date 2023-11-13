import torch, transformers
from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import unicodedata


class EndpointHandler:
    def __init__(self, path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto", torch_dtype=torch.bfloat16, load_in_4bit=True
        )
        self.pipeline = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, str]:
        # process input
        inputs = data.pop("inputs", data)

        # default parameters
        parameters = {
            "max_new_tokens": 128,
            "do_sample": True,
            "top_k": 10,
            "temperature": 1.0,
            "return_full_text": False,
        }

        # user parameters
        parameters.update(data.pop("parameters", {}))

        unique = isinstance(inputs, str)
        inputs, denormalize_funcs = claire_text_preproc_conversation(inputs)

        sequences = self.pipeline(inputs, **parameters)

        if unique:
            return [{"generated_text": denormalize_funcs(sequences[0]["generated_text"])}]
        else:
            assert len(denormalize_funcs) == len(sequences)
            return [{"generated_text": denormalize_func(seq[0]["generated_text"])} for denormalize_func, seq in zip(denormalize_funcs, sequences)]


def claire_text_preproc_conversation(text):
    if isinstance(text, (list, tuple)):
        assert len(text)
        # Apply and transpose
        texts, denormalize_funcs = zip(*[claire_text_preproc_conversation(t) for t in text])
        return list(texts), list(denormalize_funcs)

    if not isinstance(text, str):
        return text

    text = format_special_characters(text)

    text = re.sub(" - | -$|^- ", " ", text.strip(" "))

    global _reverse_tag_transfo
    _reverse_tag_transfo = {}
    text = format_special_tags(text)

    text = collapse_whitespaces_conversations(text)

    if _reverse_tag_transfo:
        reverse_tag_transfo = _reverse_tag_transfo.copy()
        def denormalize_func(t):
            for k, v in reverse_tag_transfo.items():
                if k in t:
                    t = t.replace(k, v)
            return t

        return text, lambda x: denormalize_func(x)

    else:
        return text, lambda x: x


_brackets = re.compile(r"\[([^\]]*)\]")
_pattern_speaker = re.compile(r"[^\]]+:")

# Global variable to remember some normalizations that were done and apply it back
_reverse_tag_transfo = {}
_anonymized_prefix = None


def format_special_tags(text):
    global _reverse_tag_transfo, _anonymized_prefix
    _anonymized_prefix = None
    text = re.sub(_brackets, _format_special_tags, text)
    # At last the generic anonymization
    if _anonymized_prefix:
        _reverse_tag_transfo["[Intervenant "] = _anonymized_prefix
    return text


def _format_special_tags(match):
    content_within_brackets = match.group(1)
    if re.match(_pattern_speaker, content_within_brackets):
        return _format_tag(match.group())
    else:
        return ""

def _format_tag(text):
    global _reverse_tag_transfo, _anonymized_prefix
    if text.endswith(":]"):
        anonymized_spk_prefixes = ["speaker", "spk", "locuteur"]
        # Conversion "[speaker001:]" -> "[Intervenant 1:]"
        for prefix in anonymized_spk_prefixes:
            if text.lower().startswith("["+prefix):
                try:
                    index = int(text[len(prefix)+1:-2])
                except ValueError:
                    return text
                new_spk_tag = f"[Intervenant {index}:]"
                _reverse_tag_transfo[new_spk_tag] = text
                if _anonymized_prefix is None:
                    prefix = "["+prefix
                    while len(prefix) < len(text) and text[len(prefix)] in " 0":
                        prefix += text[len(prefix)]
                    _anonymized_prefix = prefix
                return "\n" + new_spk_tag

        # Capitalize speaker name
        speaker = text[1:-2]
        speaker = capitalize(speaker)
        new_spk_tag = f"[{speaker}:]"
        if text != new_spk_tag:
            _reverse_tag_transfo[new_spk_tag] = text
        return "\n" + new_spk_tag

    # if text == "[PII]":
    #     return "[Nom]"
    # if text == "[NOISE]":
    #     return "[bruit]"
    # if text == "[LAUGHTER]":
    #     return "[rire]"
    
    return ""


def capitalize(text):
    # Custom capitalization for first and last names
    words = text.split(" ")
    words = [w.capitalize() if (not w.isupper() or len(w) > 2) else w for w in words]
    for i, w in enumerate(words):
        for sep in "-", "'":
            if sep in w:
                words[i] = sep.join(
                    [x.capitalize() if not x.isupper() else x for x in w.split(sep)]
                )
    return " ".join(words)


def collapse_whitespaces_conversations(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n ", "\n", text)
    text = re.sub(r" ([\.,])", r"\1", text)
    return text.lstrip().rstrip(" ")


def format_special_characters(text):
    text = unicodedata.normalize("NFC", text)
    for before, after in [
        ("…", "..."),
        (r"[«“][^\S\r\n]*", '"'),
        (r"[^\S\r\n]*[»”″„]", '"'),
        (r"(``|'')", '"'),
        (r"[’‘‛ʿ]", "'"),
        ("‚", ","),
        (r"–", "-"),
        ("[  ]", " "),  # unbreakable spaces
        (r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", ""),  # non-printable characters
        # ("·", "."),
        (r"ᵉʳ", "er"),
        (r"ᵉ", "e"),
    ]:
        text = re.sub(before, after, text)

    return text
