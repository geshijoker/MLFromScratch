# -*- coding: utf-8 -*-

import gradio as gr
import transformers
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

import warnings
warnings.filterwarnings('ignore')

model_id = "geshijoker/nanoLlava-dpo"

# create model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True)

# text prompt
prompt = 'Tell me what catches your eye in the image, and describe those elements in depth.'

messages = [
    {"role": "user", "content": f'<image>\n{prompt}'}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(text)

text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0)

# image, sample images can be found in images folder
image = Image.open('./test_img1.jpg')
image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)

# generate
output_ids = model.generate(
    input_ids,
    images=image_tensor,
    max_new_tokens=256,
    use_cache=True)[0]

print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())

question1 = "Tell me what catches your eye in the image, and describe those elements in depth."
image1 = Image.open('./test_img1.jpg')
question2 = "What are the main elements in this image? Describe them thoroughly."
image2 = Image.open('./test_img2.jpg')

def predict(image, question):
    # Process the question
    messages = [
        {"role": "user", "content": f'<image>\n{question}'}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0)

    # Process the image
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)

    # Generate
    output_ids = model.generate(
        input_ids.to(model.device),
        images=image_tensor.to(model.device),
        max_new_tokens=512,
        use_cache=True)[0]
    answer = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    return answer

ans = predict(image1, question1)

ans

title = "Ask mini VLM a Question"
description = """
Upload an image and ask a question related to the image. The AI was DPO finetuned to answer questions!
"""

# Define the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Question")],
    outputs=gr.Textbox(label="Answer"),
    title=title,
    description=description,
    examples=[[image1], [image2]],
)

# Launch the interface
interface.launch(share=True)

