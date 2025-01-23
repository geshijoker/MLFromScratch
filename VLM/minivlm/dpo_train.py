from datasets import load_dataset
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image
import warnings

from trl import DPOConfig, DPOTrainer
from peft import LoraConfig
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda")

dpo_dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train")
def is_grayscale(image):
    return image.mode == 'L'
dpo_dataset = dpo_dataset.filter(lambda example: not is_grayscale(example["image"]))

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True)

processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)

def format(example):
    # Prepare the input for the chat template
    prompt = [
        {
            "role": "user",
            "content": f'<image>\n{example["question"]}' #[{"type": "image"}, {"type": "text", "text": example["question"]}],
        },
    ]
    chosen = [
        {
            "role": "assistant",
            "content": example["chosen"] #[{"type": "text", "text": example["chosen"]}],
        },
    ]
    rejected = [
        {
            "role": "assistant",
            "content": example["rejected"] # [{"type": "text", "text": example["rejected"]}],
        },
    ]
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    chosen = tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=True)
    rejected = tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=True)
    # Resize the image to ensure it fits within the maximum allowable
    return {"images": example["image"], "prompt": prompt, "chosen": chosen, "rejected": rejected}

dataset = dpo_dataset.map(format, remove_columns=dpo_dataset.column_names)
# Split the dataset into train and validation
split_ratio = 0.1  # 10% for validation
train_test = dataset.train_test_split(test_size=split_ratio, seed=42)
# Access the splits
train_dataset = train_test['train']
val_dataset = train_test['test']

# Train the model
training_args = DPOConfig(
    output_dir="nanoLlava-dpo",
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    num_train_epochs=1,
    dataset_num_proc=32,  # tokenization will use 32 processes
    dataloader_num_workers=32,  # data loading will use 32 workers
    logging_steps=10,
    label_smoothing=0.1,
    push_to_hub=True,
)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = DPOTrainer(
    model,
    ref_model=None,  # not needed when using peft
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

trainer.train()
