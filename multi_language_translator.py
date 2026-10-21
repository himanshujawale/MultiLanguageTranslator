import torch
import gradio as gr
import json

from transformers import pipeline

model_path = ("../Models/models--facebook--nllb-200-distilled-600M/snapshots"
              "/f8d333a098d19b4fd9a8b18f94170487ad3f821d")
# pipe = pipeline("translation", model="facebook/nllb-200-distilled-600M")

text_translator = pipeline("translation", model=model_path,
                           torch_dtype=torch.bfloat16)



# load the json file
with open("../Files/language.json", "r") as file:
    language_data =json.load(file)

def get_FLORES_code_from_language(language):
    for entry in language_data:
        if entry['Language'].lower() == language.lower():
            return entry['FLORES-200 code']
    return None


def translate_text(text, destination_language):
    # text = "hello friends, how are you?"
    dest_code =get_FLORES_code_from_language(destination_language)
    translation = text_translator(text,
                                  src_lang="eng_Latn",
                                  tgt_lang=dest_code)
    return translation[0]["translation_text"]
    # print("Destination language:", destination_language)
    # print("FLORES code:", dest_code)
gr.close_all()

demo = gr.Interface(fn=translate_text,
    inputs=[gr.Textbox(label="Input text to translate", lines=6),gr.Dropdown(["German","French","Hindi"],label="select destination language")],
    outputs=[gr.Textbox(label="Translated text",lines=4)],
    title="GENAI PROJECT4: Multi Language Translator",
    description="THIS APPLICATION WILL BE USED TO TRANSLATE ENGLISH TEXT TO MULTI LANGUAGE")

demo.launch()




