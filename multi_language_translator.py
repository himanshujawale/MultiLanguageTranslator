import os
import json
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "../Models/models--facebook--nllb-200-distilled-600M/snapshots"
                                    "/f8d333a098d19b4fd9a8b18f94170487ad3f821d")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=dtype).to(device)
model.eval()

# load the json file
with open(os.path.join(BASE_DIR, "../Files/language.json"), "r") as file:
    language_data = json.load(file)


def get_FLORES_code_from_language(language):
    for entry in language_data:
        if entry['Language'].lower() == language.lower():
            return entry['FLORES-200 code']
    return None


def translate_text(text, destination_language):
    dest_code = get_FLORES_code_from_language(destination_language)
    if dest_code is None:
        return f"Language code not found for: {destination_language}"

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(dest_code),
            max_new_tokens=256,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


gr.close_all()

demo = gr.Interface(
    fn=translate_text,
    inputs=[gr.Textbox(label="Input text to translate", lines=6),
            gr.Dropdown(["German", "French", "Hindi"], label="select destination language")],
    outputs=[gr.Textbox(label="Translated text", lines=4)],
    title="GENAI PROJECT4: Multi Language Translator",
    description="THIS APPLICATION WILL BE USED TO TRANSLATE ENGLISH TEXT TO MULTI LANGUAGE")

demo.launch()




