# -*- coding: utf-8 -*-

import gradio as gr
import transformers
from transformers import pipeline
import torch

model_id = "geshijoker/HealthCareMagic_sft_llama3_instruct_full"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def predict(question):
    messages = [
        {"role": "system", "content": "You are a doctor, please answer the medical questions based on the patient's description."},
        {"role": "user", "content": question},
    ]
    # completion = pipe(prompt)[0]["generated_text"]
    completion = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
    )
    return completion[0]["generated_text"][-1]["content"]

question = "I fell down three stairs on my buttocks, when I got up I almost passed out I ran to my bed and waited till I felt better. I have been feeling nauseated since then and cant stand for a long period of time without feeling like im going to be sick. It is also painful to sit. Im on blood thinners what should I do?"
print(predict(question))

q1 = "hi, i have just got bitten to my year old dog 10 min ago. this morning he was seeming like he was panting and panting and breathing very heavily. We gave him water but he still did it. Now the bite feels tingily and its like its beating.I already put poroxide on. Do you know if this is normal or what to do?"
q2 = "My 3 year old daughter complains headache and throat pain since yesterday. Shes also having a hard time sleeping right now. She doesnt have fever but shes asking for blanket, telling me shes cold but were in a tropical-hot country and its really humid-hot right now, so Im really concerned if shes alright."

title = "Ask ChatDoctor a Question"
description = """
The bot was trained to answer questions based on HealthCare Magic!
"""

gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title=title,
    description=description,
    examples=[[q1], [q2]],
).launch(share=True)

