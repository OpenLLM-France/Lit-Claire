import re
import random

##############################
# Text normalization and augmentation functions

SPECIALS_TO_KEEP = [
    "PII",
    "NOISE",
    "LAUGHTER",
]

PATTERN_SPECIAL = re.compile(r"\[([^\]]*)\]")
PATTERN_SPECIAL_NOSPEAKER = re.compile(r"\[([^\]]*[^:])\]")
PATTERN_SPEAKER = re.compile(r"[^\]]+:")
PATTERN_SPEAKER_COMPLETE = re.compile(r"\[" + PATTERN_SPEAKER.pattern + r"\]")
PATTERN_SPEAKER_UNANONYMIZED = re.compile(r"\[(?!speaker\d+:)([^]]+):]")
PATTERN_PUNCTUATIONS = re.compile(r"[,\.!?…]")

def format_text(text, keep_specials=True):
    if keep_specials:
        text = re.sub(PATTERN_SPECIAL, _remove_all_except_specials, text)
    else:
        text = re.sub(PATTERN_SPECIAL, _remove_all_except_speakers_and_pii, text)
    return collapse_whitespaces(text)
    
def _remove_all_except_specials(match):
    content_within_brackets = match.group(1)
    if re.match(PATTERN_SPEAKER, content_within_brackets) or content_within_brackets in SPECIALS_TO_KEEP:
        return match.group().lower()
    else:
        return ""
    
def _remove_all_except_speakers_and_pii(match):
    content_within_brackets = match.group(1)
    if re.match(PATTERN_SPEAKER, content_within_brackets):
        return match.group().lower()
    elif content_within_brackets == "pii":
        import names
        return names.get_first_name().lower()
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

def anonymize_speakers(text):
    # Get all speakers
    speakers = [] 
    [speakers.append(x) for x in re.findall(PATTERN_SPEAKER_COMPLETE, text) if x not in speakers] 
    new_speakers = [f"[speaker{i+1:03d}:]" for i in range(len(speakers))]
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
