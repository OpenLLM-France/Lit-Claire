import re
import random
try:
    import names
except ImportError:
    import warnings
    warnings.warn("Could not import names. Text augmentation will fail. You can install name using `pip install names`.")

##############################
# Text normalization and augmentation functions

SPECIALS_TO_KEEP = [
    "PII",
    "NOISE",
    "LAUGHTER",
]

FRANCIZISE_SPECIALS = True

PATTERN_SPEAKER = re.compile(r"[^\]]+:")
PATTERN_SPEAKER_COMPLETE = re.compile(r"\[" + PATTERN_SPEAKER.pattern + r"\]")
PATTERN_SPEAKER_UNANONYMIZED = re.compile(r"\[(?!speaker\d+:)([^]]+):]")

PATTERN_SPECIAL = re.compile(r"\[([^\]]*)\]")
PATTERN_SPECIAL_NOSPEAKER = re.compile(r"\[([^\]]*[^:])\]")
PATTERN_PUNCTUATIONS = re.compile(r"[,\.!?…]")


def format_text(text, keep_specials=True):
    if keep_specials:
        text = re.sub(PATTERN_SPECIAL, _remove_all_except_specials, text)
    else:
        text = re.sub(PATTERN_SPECIAL, _remove_all_except_speakers_and_pii, text)
    return collapse_whitespaces(text)

if FRANCIZISE_SPECIALS:

    def format_special(text):
        if text.endswith(":]"):
            if text.startswith("[speaker"):
                # "[speaker001:]" -> "[Intervenant 1:]"
                index = int(text[8:11])
                return f"[Intervenant {index}:]"
            else:
                # "[claude-marie Claude-Marie JR:]" -> "[Claude-Marie Claude-Marie JR:]"
                speaker = capitalize(text[1:-2])
                return f"[{speaker}:]"
        if text == "[PII]":
            return "[Nom]"
        if text == "[NOISE]":
            return "[bruit]"
        if text == "[LAUGHTER]":
            return "[rire]"

    def speaker_tag(i):
        return f"[Locuteur {i+1}:]"

else:

    def format_special(text):
        return text.lower()
    
    def speaker_tag(i):
        return f"[speaker{i+1:03d}:]"

def _remove_all_except_specials(match):
    content_within_brackets = match.group(1)
    if re.match(PATTERN_SPEAKER, content_within_brackets) or content_within_brackets in SPECIALS_TO_KEEP:
        return format_special(match.group())
    else:
        return ""
    
def _remove_all_except_speakers_and_pii(match):
    content_within_brackets = match.group(1)
    if re.match(PATTERN_SPEAKER, content_within_brackets):
        return format_special(match.group())
    elif content_within_brackets == "pii":
        return names.get_first_name()
    else:
        return ""

def collapse_whitespaces(text):
    text = re.sub(r" +", " ", text)
    text = re.sub(r" ([\.,])", r"\1", text)
    return text.strip()

def remove_punctuations(text):
    text = re.sub(PATTERN_PUNCTUATIONS, "", text)
    return collapse_whitespaces(text)

def to_lower_case(text):
    return text.lower()

def capitalize(text):
    # michel JR claude-marie -> Michel JR Claude-Marie
    words = text.split(" ")
    words = [w.capitalize() if not w.isupper() else w for w in words]
    for i, w in enumerate(words):
        if "-" in w:
            words[i] = "-".join([x.capitalize() if not x.isupper() else x for x in w.split("-")])
    return " ".join(words)

def anonymize_speakers(text):
    # Get all speakers
    speakers = [] 
    [speakers.append(x) for x in re.findall(PATTERN_SPEAKER_COMPLETE, text) if x not in speakers] 
    new_speakers = [speaker_tag(i) for i in range(len(speakers))]
    for spk, nspk in zip(speakers, new_speakers):
        text = text.replace(spk, nspk)
    return text

def unanonymize_speakers(text):
    # Get all speakers
    speakers = [] 
    [speakers.append(x) for x in re.findall(PATTERN_SPEAKER_COMPLETE, text) if x not in speakers] 
    if random.random() < 0.5:
        # Use first names only
        new_speakers = [f"[{names.get_first_name()}:]" for i in range(len(speakers))]
    else:
        # Use first and last name
        new_speakers = [f"[{names.get_first_name()} {names.get_last_name()}:]" for i in range(len(speakers))]
    for spk, nspk in zip(speakers, new_speakers):
        text = text.replace(spk, nspk)
    return text

def has_upper_case(text):
    return bool(re.search(r"[A-Z]", text))

def has_speaker_id(text):
    return bool(re.search(PATTERN_SPEAKER_UNANONYMIZED, text))

def has_punctuation(text):
    return bool(re.search(PATTERN_PUNCTUATIONS, text))

def has_specials(text):
    return bool(re.search(PATTERN_SPECIAL_NOSPEAKER, text))

def augmented_texts_generator(text, max_variants=2, force_augmentation=False):
    """
    Generate several variants of a text.
    max_variants: maximum number of variants returned
    force_augmentation: if True, when max_variants=0, return a random augmentation
    """
    if max_variants == 0 and force_augmentation:
        all_variants = list(augmented_texts_generator(text, max_variants=4))
        random.shuffle(all_variants)
        yield all_variants[0]
        return

    text = format_text(text)
    yield text
    if max_variants <= 0:
        return

    _specials = has_specials(text)
    _speaker = has_speaker_id(text)
    _upper = has_upper_case(text)
    _punct = has_punctuation(text)

    coming_next = _boole(_specials) + _boole(_speaker) + _boole(_upper) + _boole(_punct)

    if _specials:
        text = format_text(text, keep_specials=False)
        if max_variants > coming_next-1:
            yield text
        coming_next -= 1
    if _speaker:
        text = anonymize_speakers(text)
        if max_variants > coming_next-1:
            yield text
        coming_next -= 1
    else:
        # Sometimes unanonymize speakers
        r = random.random()
        if r < 0.2:
            if max_variants > coming_next-1:
                text2 = unanonymize_speakers(text)
                yield text2
            coming_next -= 1
    if _upper:
        text = to_lower_case(text)
        if max_variants > coming_next-1:
            yield text
        coming_next -= 1
    if _punct:
        text = remove_punctuations(text)
        if max_variants > coming_next-1:
            yield text
        coming_next -= 1

def _boole(x):
    return 1 if x else 0

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Test text normalization and augmentation.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("words", type=str, nargs="+", help="Some text")
    parser.add_argument("--max_variants", type=int, default=4, choices=[0, 1, 2, 3, 4], help="Augmentation max_variants.")
    args = parser.parse_args()

    text = " ".join(args.words)
    max_variants = args.max_variants

    INCLUDE_LINE_BREAKS = False
    if INCLUDE_LINE_BREAKS:
        text = re.sub(r" ("+PATTERN_SPEAKER_COMPLETE.pattern+r")", r"\n\1", text)
    def format_stdout(text):
        return text.replace("\n", "\\n")
    print("Original      :", format_stdout(text))
    # print("Normalized (2):", format_stdout(format_text(text, keep_specials=False)))
    for ivariant, text_variant in enumerate(augmented_texts_generator(text, max_variants)):
        print(f"Augmented ({ivariant}/{max_variants}):" if ivariant > 0 else "Normalized     :", format_stdout(text_variant))

