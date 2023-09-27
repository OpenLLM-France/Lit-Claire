import re

##############################
# Text normalization and augmentation functions

SPECIALS_TO_KEEP = [
    "PII",
    "NOISE",
    "LAUGHTER",
]

PATTERN_SPECIAL = re.compile(r"\[([^\]]*)\]")
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
    if re.match(PATTERN_SPEAKER, content_within_brackets) or content_within_brackets == "PII":
        return match.group().lower()
    else:
        return ""

def collapse_whitespaces(text):
    return re.sub(r" +", " ", text).strip()

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

def augmented_texts_generator(text):
    text = format_text(text)
    yield text
    _upper = has_upper_case(text)
    _speaker = has_speaker_id(text)
    _punct = has_punctuation(text)
    if _speaker:
        text_anonym = anonymize_speakers(text)
        yield text_anonym
    if _upper:
        yield to_lower_case(text)
        if _speaker:
            yield to_lower_case(text_anonym)
    if _punct:
        text_no_punct = remove_punctuations(text)
        yield text_no_punct
        if _upper:
            yield to_lower_case(text_no_punct)
            if _speaker:
                yield remove_punctuations(to_lower_case(text_anonym))
        if _speaker:
            yield remove_punctuations(text_anonym)

if __name__ == "__main__":
    
    import sys
    text = " ".join(sys.argv[1:])

    INCLUDE_LINE_BREAKS = False
    if INCLUDE_LINE_BREAKS:
        text = re.sub(r" ("+PATTERN_SPEAKER_COMPLETE.pattern+r")", r"\n\1", text)
    def format_stdout(text):
        return text.replace("\n", "\\n")
    print("Original      :", format_stdout(text))
    print("Normalized (2):", format_stdout(format_text(text, keep_specials=False)))
    for ivariant, text_variant in enumerate(augmented_texts_generator(text)):
        print(f"Augmented ({ivariant}):" if ivariant > 0 else "Normalized   :", format_stdout(text_variant))

