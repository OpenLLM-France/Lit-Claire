import gradio as gr
from threading import Thread
import transformers
import spaces
import torch
import unicodedata
import regex as re

# Model
model_name = "OpenLLM-France/Claire-7B-0.1"

# Title and description
title = "Conversation avec Claire"
description = """\
Simulation de conversation en Français avec [OpenLLM-France/Claire-7B](https://huggingface.co/OpenLLM-France/Claire-7B-0.1).
<strong>Claire n'est <u>pas</u> un assistant personnel</strong>, elle a tendance à comprendre et répondre un <b>langage parlé</b>, \
peut faire preuve d'humour, et <strong>ne vous dira <u>pas</u> (forcément) des vérités</strong>.
"""

# Default variables
default_max_new_tokens = 200
default_temperature = 1.0
default_repetition_penalty = 1.5
default_top_k = 10
default_top_p = 0.99

default_parameters = [
    default_max_new_tokens,
    default_temperature,
    default_repetition_penalty,
    default_top_k,
    default_top_p,
]

# Examples
examples = [
    [
        "Bonjour Claire. Quel est votre sport préféré?",  # user_message
        False,
        "",  # bot_message_start
        # "",  # First name
        *default_parameters,
    ],
    [
        "Bonjour. Je vous propose de faire un tour de table.",  # user_message
        True, # more than one turn
        "",  # bot_message_start
        # "",  # First name
        *default_parameters,
    ],
    [
        "Que vas-tu nous cuisiner aujourd'hui?",  # user_message
        False,
        "Alors, nous allons voir la recette",  # bot_message_start
        # "", # First name
        *default_parameters,
    ],
]

# Override default gradio buttons
gradio_buttons = dict(
    submit_btn=gr.Button("Envoyer"), # Sumbit
    retry_btn=gr.Button("🔄  Générer une autre réponse"), # "🔄  Retry"
    undo_btn=gr.Button("↩️ Annuler"), # "↩️ Undo"
    clear_btn=gr.Button("🗑️  Effacer la conversation"), # "🗑️  Clear"
    # stop_btn= None,
    stop_btn=gr.Button("Arrêter"), # Stop
)
additional_inputs_name="Paramètres" # "Additional inputs"
textbox=gr.Textbox(
    container=False,
    show_label=False,
    label="Message",
    placeholder="Votre message (laissez vide pour que le Bot continue seul)...",
    scale=7,
    lines=2,
    autofocus=False,
)
chatbot_label="Conversation" # Chatbot


additional_inputs = [
    gr.Checkbox(
        False,
        label="Plus qu'un tour de parole",
        info="Générer plusieurs tours de parole (et donc comment vous pourriez continuer la conversation)",
    ),
    gr.Textbox(
        "",
        label="Début de réponse",
        info="Vous pouvez taper ici ce que commence à vous répondre le Bot (pensez à actualiser entre chaque génération)",
        type="text",
    ),
    # gr.Textbox(
    #     "",
    #     label="Votre prénom",
    #     info="Prénom de vous en tant qu'interlocuteur (si vous vous nommez, le bot s'appellera Claire)",
    # ),
    gr.Slider(
        label="Longueur max",
        info="Longueur maximale du texte généré (en nombre de 'tokens' ~ mots et ponctuations)",
        value=default_max_new_tokens,
        minimum=25,
        maximum=1000,
        step=25,
        interactive=True,
    ),
    gr.Slider(
        label="Température",
        info="Une valeur élevée augmente la diversité du texte généré, mais peut aussi produire des résultats incohérents",
        value=default_temperature,
        minimum=0.1,
        maximum=1.9,
        step=0.1,
        interactive=True,
    ),
    gr.Slider(
        label="Pénalité de répétition",
        info="Pénalisation des répétitions",
        value=default_repetition_penalty,
        minimum=1.0,
        maximum=1.95,
        step=0.05,
        interactive=True,
    ),
    gr.Slider(
        label="Top-k",
        info="Une valeur élevée permet d'explorer plus d'alternatives",
        value=default_top_k,
        minimum=1,
        maximum=50,
        step=1,
        interactive=True,
    ),
    gr.Slider(
        label="Top-p",
        info="Une valeur élevée permet d'explorer plus d'alternatives",
        value=default_top_p,
        minimum=0.9,
        maximum=1.0,
        step=0.01,
        interactive=True,
    ),
]

STREAMING = True

print("Loading model...")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
)

# print("Optimizing model...")
# import optimum
# from optimum.bettertransformer import BetterTransformer
# model = BetterTransformer.transform(model)

print("Setup chat...")

eos_token_id = tokenizer.eos_token_id
newspk_token_id = tokenizer.encode("[")
assert len(newspk_token_id) == 1
newspk_token_id = newspk_token_id[0]

tokenizer.add_special_tokens({"eos_token": "["})

user_internal_tag = "[Intervenant 1:]"
bot_internal_tag = "[Intervenant 2:]"
device = "cuda" if torch.cuda.is_available() else "cpu"


@spaces.GPU
def generate(
    user_message,
    conversation_history=[],
    generate_several_turns=False,
    bot_message_start="",
    # user_surname="",
    max_new_tokens=default_max_new_tokens,
    temperature=default_temperature,
    repetition_penalty=default_repetition_penalty,
    top_k=default_top_k,
    top_p=default_top_p,
    user_surname="", # Experimental (TODO)
    remove_unfinished_sentence=True,
):
    user_message = claire_text_preproc_message(user_message)
    bot_message_start = claire_text_preproc_message(bot_message_start)

    if user_surname:
        user_surname = capitalize(collapse_whitespaces(re.sub(r"[^\p{L}\-\.']", " ", user_surname))).strip()
    if user_surname:
        user_tag = f"[{user_surname}:]"
        bot_tag = f"[Claire:]"
    else:
        user_tag = user_internal_tag
        bot_tag = bot_internal_tag

    if conversation_history:
        conversation_history = "\n".join(
            [
                f"{user_tag} {claire_text_preproc_message(user)}\n{bot_tag} {claire_text_preproc_message(bot) if bot else ''}"
                for user, bot in conversation_history
            ]
        )
        conversation_history = from_display_to_internal(conversation_history)
        conversation_history = conversation_history.rstrip()
        if conversation_history:
            conversation_history += "\n"
    else:
        conversation_history = ""
    if not bot_message_start:
        bot_message_start = ""

    # Combine the user and bot messages into a conversation
    conversation = f"{conversation_history}{user_tag} {user_message}\n{bot_tag} {bot_message_start}".strip()
    conversation = remove_empty_turns(conversation)

    # Encode the conversation using the tokenizer
    input_ids = tokenizer.encode(
        conversation, return_tensors="pt", add_special_tokens=True
    )
    input_ids = input_ids.to(device)

    skip_special_tokens = not generate_several_turns

    if STREAMING:
        streamer = transformers.TextIteratorStreamer(
            tokenizer,
            timeout=10.0,
            skip_prompt=True,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    else:
        streamer = None

    # Generation parameters
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        eos_token_id=eos_token_id if generate_several_turns else newspk_token_id,
        pad_token_id=eos_token_id,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        num_beams=1,
        # use_cache=False,
        # early_stopping=False,
    )
    if STREAMING:
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        if bot_message_start.strip():
            yield bot_message_start
        for token in streamer:
            # Ignore line breaks
            if not generate_several_turns and re.match(r"\s*\n$", token):
                continue
            outputs.append(token)
            text = bot_message_start + from_internal_to_display("".join(outputs))
            yield text
    else:
        output_ids = model.generate(**generate_kwargs)
        output_ids = output_ids[0][len(input_ids[0]) :]
        text = tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)
        if bot_message_start.strip():
            bot_message_start = bot_message_start.strip() + " "

        text = bot_message_start + from_internal_to_display(text.rstrip("[").strip())
        yield text

    if generate_several_turns:
        if remove_unfinished_sentence:
            yield remove_last_unfinished_sentence(text)
        else:
            yield remove_last_unfinished_turn(text)[0]


def claire_text_preproc_message(text):
    text = format_punctuations_for_french(text)
    text = format_special_characters(text)
    text = collapse_whitespaces(text)
    text = replace_brackets(text)
    return text


def collapse_whitespaces(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r" ([\.,])", r"\1", text)
    return text.lstrip().rstrip(" ")


def replace_brackets(text):
    text = re.sub(r"[\[\{]", "(", text)
    text = re.sub(r"[\]\}]", ")", text)
    return text

def format_punctuations_for_french(text):
    for before, after in french_punctuation_rules:
        text = re.sub(before, after, text)
    return text

french_punctuation_rules = {
    # Add a space before double punctuation marks
    (r"([" + re.escape('?!:;') + r"])", r" \1"),
    # Remove space before simple punctuation marks
    (r"\s+([" + re.escape(',.') + r"])", r"\1"),
    # Add space after punctuation marks
    (r"([" + re.escape('?!:;,') + r"]+)([^ " + re.escape('?!:;,') + r"\d])", r"\1 \2"),
    (r"([" + re.escape('.') + r"]+)([A-Z])", r"\1 \2"),
}

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


user_name = "[Vous:]"
bot_name = "[Bot:]"


def from_internal_to_display(text):
    for before, after in [
        (user_internal_tag, user_name),
        (bot_internal_tag, bot_name),
    ]:
        text = text.replace(before, after)
    return text


def from_display_to_internal(text):
    for before, after in [
        (user_name, user_internal_tag),
        (bot_name, bot_internal_tag),
    ]:
        text = text.replace(before, after)
    return text


def remove_last_unfinished_sentence(text):
    text, removed_turn = remove_last_unfinished_turn(text)
    if removed_turn:
        return text
    line_breaks = [u.span(0)[0] for u in re.finditer("\n", text)]
    remove_last_sentence = True
    if len(line_breaks) >= 1 and len(text[line_breaks[-1]:].split("]")[-1]) < 15:
        text = text[: line_breaks[-1]]
        line_breaks.pop(-1)
        remove_last_sentence = False
    if remove_last_sentence and len(line_breaks) >= 1:
        sentence_ends = [u.span(0)[0] for u in re.finditer(r"[\.!?]", text)]
        sentence_ends = [p for p in sentence_ends if p > line_breaks[-1]]
        if sentence_ends:
            text = text[: sentence_ends[-1] + 1]
        else:
            phrase_ends = [u.span(0)[0] for u in re.finditer(r"[,;]", text)]
            phrase_ends = [p for p in phrase_ends if p > line_breaks[-1]]
            if phrase_ends:
                text = text[: phrase_ends[-1] + 1]
    return text


def remove_last_unfinished_turn(text):
    starts = [u.span(0)[0] for u in re.finditer(r"\[", text)]
    did_it = False
    if starts and "]" not in text[starts[-1] :]:
        text = text[: starts[-1]]
        did_it = True
    return text.rstrip(), did_it


def remove_empty_turns(text):
    while re.search(_empty_turn, text):
        # Remove empty turns
        text = re.sub(_empty_turn, r"\1", text)
        # Remove same speaker speaking twice
        text = re.sub(_repeated_turn, r"\1 \2", text)
    return text

_speaker_regex = r"\[[^\]]+:\]"
_empty_turn = re.compile(_speaker_regex + r"[^\p{L}]*" + "(" + _speaker_regex + ")")
_repeated_turn = re.compile(r"(" + _speaker_regex + r") ([^\[]*)\s\1")


def capitalize(text):
    # michel JR claude-marie -> Michel JR Claude-Marie
    words = text.split(" ")
    words = [w.capitalize() if (not w.isupper() or len(w)>2) else w for w in words]
    for i, w in enumerate(words):
        for sep in "-", "'":
            if sep in w:
                words[i] = sep.join([x.capitalize() if not x.isupper() else x for x in w.split(sep)])
    return " ".join(words)

# # Test
# list(generate(*(examples[0][:1] + [[]] + examples[0][1:])))


chat_interface = gr.ChatInterface(
    fn=generate,
    title=title,
    description=description,
    chatbot=gr.Chatbot(label=chatbot_label),
    textbox=textbox,
    examples=examples,
    additional_inputs=additional_inputs,
    additional_inputs_accordion=gr.Accordion(
        label="Paramètres",
        open=True,
    ),
    autofocus=False,
    **gradio_buttons,
)

if __name__ == "__main__":
    print("Launching chat...")
    with gr.Blocks(css="style.css") as demo:
        chat_interface.render()
        demo.queue(max_size=20).launch()
