import regex as re
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
USE_DASHES = True

PATTERN_SPEAKER_INBRACKETS = re.compile(r"[^\]]+:")
PATTERN_SPEAKER = re.compile(r"\[" + PATTERN_SPEAKER_INBRACKETS.pattern + r"\]")
PATTERN_SPEAKER_UNANONYMIZED = re.compile(r"\[(?!speaker\d+:)([^]]+):]")

PATTERN_SPECIAL = re.compile(r"\[([^\]]*)\]")
PATTERN_SPECIAL_NOSPEAKER = re.compile(r"\[([^\]]*[^:])\]")

PATTERN_PUNCTUATIONS = re.compile(r"[,;\.!?…]|: ")

def format_text(text, keep_specials=True):
    if keep_specials:
        text = re.sub(PATTERN_SPECIAL, _remove_all_except_specials, text)
    else:
        text = re.sub(PATTERN_SPECIAL, _remove_all_except_speakers_and_pii, text)
    text = remove_empty_turns(text)
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
                speaker = text[1:-2]
                speaker = capitalize(speaker)
                assert re.match(r"[A-ZÉÈËÊÔÀÁÂ0-9]", speaker), f"Unexpected speaker {text} ({[ord(c) for c in text[:6]]}) -> {speaker}"
                return f"[{speaker}:]"
        if text == "[PII]":
            return "[Nom]"
        if text == "[NOISE]":
            return "[bruit]"
        if text == "[LAUGHTER]":
            return "[rire]"

    def speaker_tag(i):
        return f"[Intervenant {i+1}:]"

else:

    def format_special(text):
        return text.lower()
    
    def speaker_tag(i):
        return f"[speaker{i+1:03d}:]"

def _remove_all_except_specials(match):
    content_within_brackets = match.group(1)
    if re.match(PATTERN_SPEAKER_INBRACKETS, content_within_brackets) or content_within_brackets in SPECIALS_TO_KEEP:
        return format_special(match.group())
    else:
        return ""
    
def _remove_all_except_speakers_and_pii(match):
    content_within_brackets = match.group(1)
    if re.match(PATTERN_SPEAKER_INBRACKETS, content_within_brackets):
        return format_special(match.group())
    elif content_within_brackets in ["Nom", "nom", "PII", "pii"]:
        return names.get_first_name()
    else:
        return ""

def collapse_whitespaces(text):
    text = re.sub(r"[\t ]+", " ", text)
    # Remove extra spaces that could appear when removing special tags
    text = re.sub(r"(\w) ([\.,])", r"\1\2", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()

def remove_punctuations(text):
    text = re.sub(PATTERN_PUNCTUATIONS, "", text)
    return collapse_whitespaces(text)

def to_lower_case(text):
    """ Lowercase all except text within brackets """
    pattern = r'(\[[^\]]*\])|([^\[]+)'
    result = re.sub(pattern, _lowercase_except_brackets, text)
    return result

def _lowercase_except_brackets(match):
    text = match.group(0)
    if text.startswith("["):
        return text
    return text.lower()


def capitalize(text):
    # michel JR claude-marie -> Michel JR Claude-Marie
    words = text.split(" ")
    words = [w.capitalize() if (not w.isupper() or len(w)>2) else w for w in words]
    for i, w in enumerate(words):
        for sep in "-", "'":
            if sep in w:
                words[i] = sep.join([x.capitalize() if not x.isupper() else x for x in w.split(sep)])
    return " ".join(words)

def anonymize_speakers(text):
    # Get all speakers
    speakers = [] 
    [speakers.append(x) for x in re.findall(PATTERN_SPEAKER, text) if x not in speakers] 
    new_speakers = [speaker_tag(i) for i in range(len(speakers))]
    for spk, nspk in zip(speakers, new_speakers):
        text = text.replace(spk, nspk)
    return text

def unanonymize_speakers(text):
    # Get all speakers
    speakers = [] 
    [speakers.append(x) for x in re.findall(PATTERN_SPEAKER, text) if x not in speakers] 
    if random.random() < 0.5:
        # Use first names only
        new_speakers = [f"[{names.get_first_name()}:]" for i in range(len(speakers))]
    else:
        # Use first and last name
        new_speakers = [f"[{names.get_first_name()} {names.get_last_name()}:]" for i in range(len(speakers))]
    for spk, nspk in zip(speakers, new_speakers):
        text = text.replace(spk, nspk)
    return text

def dash_speakers(text):
    return re.sub(PATTERN_SPEAKER, "-", text)

def has_upper_case(text):
    return bool(re.search(r"[A-Z]", re.sub(r"\[[^]]+\]", "", text)))

def has_speaker_id(text):
    return bool(re.search(PATTERN_SPEAKER_UNANONYMIZED, text))

def has_punctuation(text):
    return bool(re.search(PATTERN_PUNCTUATIONS, text))

def has_specials(text):
    return bool(re.search(PATTERN_SPECIAL_NOSPEAKER, text))

def remove_empty_turns(text):
    if re.search(PATTERN_EMPTY_TURN, text):
        # Remove empty turns
        text = re.sub(PATTERN_EMPTY_TURN, r"\1", text)
        # Remove same speaker speaking twice
        text = re.sub(PATTERN_REPEATED_TURN, r"\1 \2", text)
    return text

PATTERN_EMPTY_TURN = re.compile(PATTERN_SPEAKER.pattern + r"[^\p{L}]*" + "("+PATTERN_SPEAKER.pattern+")")
PATTERN_REPEATED_TURN = re.compile(r"("+PATTERN_SPEAKER.pattern+r") ([^:]*)\s\1")
PATTERN_SPEAKER_LOOSE = re.compile(r"\s*"+PATTERN_SPEAKER.pattern+r"\s*")

def augmented_texts_generator(text, max_variants=4, force_augmentation=False, keep_specials=False):
    """
    Generate several variants of a text.
    max_variants: maximum number of variants returned
    force_augmentation: if True when max_variants=0, return a random augmentation (including original text normalized)
    """
    only_one_variant = (max_variants == 1)
    if not only_one_variant and (bool(max_variants) or (max_variants == 0 and force_augmentation)):
        all_variants = list(augmented_texts_generator(text, max_variants=None))
        if max_variants:
            # Provide normalized text first
            yield all_variants[0]
            all_variants = all_variants[1:]
        # Choose randomly in the rest
        random.shuffle(all_variants)
        for i in range(min(len(all_variants), max(1, max_variants))):
            yield all_variants[i]
        return

    text1 = format_text(text, keep_specials=keep_specials)
    yield text1
    if max_variants == 0:
        return

    do_specials = keep_specials and has_specials(text1)
    do_anonymize = True
    can_anonymize = has_speaker_id(text)
    do_lower_case = has_upper_case(text)
    do_remove_punc = has_punctuation(text)
    if USE_DASHES:
        num_speakers = len(set(re.findall(PATTERN_SPEAKER, text)))
        assert num_speakers
    else:
        num_speakers = 0

    if only_one_variant:
        # print(f"Original: {do_specials=} {do_anonymize=} {do_lower_case=} {do_remove_punc=} {num_speakers=}")

        do_dash = num_speakers in [1, 2]
        # Optimized path for single variant: (because processing big corpus like "Assemblée Nationale" was taking too long)
        # We randomly choose
        do_specials, do_anonymize, do_lower_case, do_remove_punc, do_dash = randomize_boolean_variables(do_specials, do_anonymize, do_lower_case, do_remove_punc, do_dash)
        if not do_dash:
            num_speakers = 0

    # print(f"Applying: {do_specials=} {do_anonymize=} {do_lower_case=} {do_remove_punc=} {num_speakers=}")

    if do_specials:
        text1 = format_text(text1, keep_specials=False)
        if not only_one_variant:
            yield text1

    if do_anonymize:
        if can_anonymize:
            text2 = anonymize_speakers(text1)
        else:
            text2 = unanonymize_speakers(text1)
        if not only_one_variant:
            yield text2
        texts = [text1, text2]
    else:
        texts= [text1]

    text3 = None
    if num_speakers == 2:
        text3 = dash_speakers(text1)
    elif num_speakers == 1:
        text3 = re.sub(PATTERN_SPEAKER_LOOSE, "", text1)
    if text3:
        if not only_one_variant:
            yield text3
            texts.append(text3)
        else:
            texts = [text3]

    for text in texts:
        has_lower_cased = False
        if (do_lower_case and do_remove_punc) and random.random() < 0.5:
            # lowercase first 
            if do_lower_case:
                text = to_lower_case(text)
                has_lower_cased = True
                if not only_one_variant:
                    yield text
        if do_remove_punc:
            text = remove_punctuations(text)
            if not only_one_variant:
                yield text
        if do_lower_case and not has_lower_cased:
            text = to_lower_case(text)
            if not only_one_variant:
                yield text
    
    if only_one_variant:
        yield text


def randomize_boolean_variables(*variables):
    variables = list(variables)

    # Create a list of indices for True variables
    true_indices = [i for i, var in enumerate(variables) if var]

    # Ends early if not True variables
    if not true_indices:
        return variables

    # Randomly choose one of the True variables to keep as True
    random.shuffle(true_indices)
    num_to_turn_on = random.randint(1, len(true_indices))
    true_to_keep = true_indices[:num_to_turn_on]

    # Randomly change the status of the other True variables to False
    for i in true_indices:
        if i not in true_to_keep:
            variables[i] = False

    return variables


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Test text normalization and augmentation.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("words", type=str, nargs="+", help="Some text")
    parser.add_argument("--max_variants", type=int, default=None, help="Augmentation max_variants.")
    parser.add_argument("--seed", type=int, default=random.randint(1, 1000), help="Random seed")
    parser.add_argument("--force_augmentation", default=False, action="store_true", help="Force augmentation even when max_variants=0.")
    args = parser.parse_args()

    import random

    text = " ".join(args.words)
    max_variants = args.max_variants

    INCLUDE_LINE_BREAKS = False
    if INCLUDE_LINE_BREAKS:
        text = re.sub(r" ("+PATTERN_SPEAKER.pattern+r")", r"\n\1", text)
    def format_stdout(text):
        return text.replace("\n", "\\n")
    print("Original      :", format_stdout(text))
    # print("Normalized (2):", format_stdout(format_text(text, keep_specials=False)))
    random.seed(args.seed)
    for ivariant, text_variant in enumerate(augmented_texts_generator(text, max_variants, force_augmentation=args.force_augmentation)):
        print(f"Augmented ({ivariant}/{max_variants}):" if ivariant > 0 else "Normalized     :", format_stdout(text_variant))

