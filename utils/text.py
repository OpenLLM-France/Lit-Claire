import re

##############################
# Text normalization functions

def collapse_whitespaces(text):
    return re.sub(r" +", " ", text).strip()

def remove_special_words(text):
    # Remove all [*] except the one at the beginning and after linebreaks
    text = re.sub(r"([^\n])\[[^\]]*\]", r"\1", text)
    return collapse_whitespaces(text)
    
def remove_punctuations(text):
    text = re.sub(r"[,\.!?…]", "", text)
    return collapse_whitespaces(text)

def to_lower_case(text):
    return text.lower()

def anonymize_speakers(text):
    # Get all speakers
    speakers = [] 
    [speakers.append(x) for x in re.findall(r"\[([^\]]+):\]", text) if x not in speakers] 
    new_speakers = [f"speaker{i+1:03d}" for i in range(len(speakers))]
    for spk, nspk in zip(speakers, new_speakers):
        text = text.replace(f"[{spk}:", f"[{nspk}:")
    return text

def has_upper_case(text):
    return bool(re.search(r"[A-Z]", text))

def has_speaker_id(text):
    return bool(re.search(r"\[[^spkeaker\d]+:\]", text))

def has_punctuation(text):
    return bool(re.search(r"[,\.!?…]", text))

def augmented_texts_generator(text):
    text = remove_special_words(text)
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