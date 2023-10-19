import torch, transformers
from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


class EndpointHandler:
    def __init__(self, path=""):
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
        inputs = claire_text_preproc(inputs)

        # default parameters
        parameters = {
            "max_length": 128,
            "do_sample": True,
            "top_k": 10,
            "temperature": 1.0,
            "return_full_text": False,
        }

        parameters.update(data.pop("parameters", {}))

        sequences = self.pipeline(inputs, **parameters)

        return [{"generated_text": sequences[0]["generated_text"]}]


def claire_text_preproc(text):
    if isinstance(text, (list, tuple)):
        return [claire_text_preproc(t) for t in text]

    if not isinstance(text, str):
        return text

    text = format_special_characters(text)

    # text = remove_ligatures(text)

    text = re.sub(" - | -$|^- ", " ", text.strip())

    text = format_special_tags(text)

    text = collapse_whitespaces(text)

    return text


_brackets = re.compile(r"\[([^\]]*)\]")
_pattern_speaker = re.compile(r"[^\]]+:")
_non_printable_pattern = r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]"


def collapse_whitespaces(text):
    text = re.sub(r" +", " ", text)
    text = re.sub(r" ([\.,])", r"\1", text)
    return text.strip()


def format_special_tags(text):
    return re.sub(_brackets, _format_special_tags, text)


def _format_special_tags(match):
    content_within_brackets = match.group(1)
    if re.match(_pattern_speaker, content_within_brackets):
        return _format_tag(match.group())
    else:
        return ""


def _format_tag(text):
    if text.endswith(":]"):
        if text.startswith("[speaker"):
            # "[speaker001:]" -> "[Intervenant 1:]"
            index = int(text[8:11])
            return f"\n[Intervenant {index}:]"
        else:
            # "[claude-marie Claude-Marie JR:]" -> "[Claude-Marie Claude-Marie JR:]"
            speaker = text[1:-2]
            speaker = capitalize(speaker)
            return f"\n[{speaker}:]"
    if text == "[PII]":
        return "[Nom]"
    if text == "[NOISE]":
        return "[bruit]"
    if text == "[LAUGHTER]":
        return "[rire]"


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


def format_special_characters(text):
    for before, after in [
        ("â", "â"),
        ("à", "à"),
        ("á", "á"),
        ("ê", "ê"),
        ("é", "é"),
        ("è", "è"),
        ("ô", "ô"),
        ("û", "û"),
        ("î", "î"),
        ("\x92", "'"),
        ("…", "..."),
        (r"[«“][^\S\r\n]*", '"'),
        (r"[^\S\r\n]*[»”″„]", '"'),
        (r"(``|'')", '"'),
        (r"[’‘‛ʿ]", "'"),
        ("‚", ","),
        (r"–", "-"),
        # non
        ("[  ]", " "),  # weird whitespace
        (_non_printable_pattern, ""),  # non-printable characters
        ("·", "."),
        (r"ᵉʳ", "er"),
        (r"ᵉ", "e"),
    ]:
        text = re.sub(before, after, text)

    return text


def remove_ligatures(text):
    text = re.sub(r"œ", "oe", text)
    text = re.sub(r"æ", "ae", text)
    text = re.sub(r"ﬁ", "fi", text)
    text = re.sub(r"ﬂ", "fl", text)
    text = re.sub("ĳ", "ij", text)
    text = re.sub(r"Œ", "Oe", text)
    text = re.sub(r"Æ", "Ae", text)
    return text
