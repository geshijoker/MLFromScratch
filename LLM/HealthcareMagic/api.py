import gradio as gr

title = "Health Care Magic Chat Doctor"
description = "Gradio Demo for Chat Doctor finetuned with Llama"

gr.Interface.load(
    "geshijoker/HealthCareMagic_sft_llama3_instruct_full",
    inputs=gr.Textbox(lines=5, label="Input Text"),
    title=title,
    description=description,
).launch()