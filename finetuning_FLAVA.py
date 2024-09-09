"""
Please check out 
    PyTorch Guide: https://pytorch.org/tutorials/beginner/flava_finetuning_tutorial.html
    HuggingFace Guide: https://huggingface.co/docs/accelerate/en/index
"""

import torch
from torch import nn
from torch import optim
from torchvision import transforms
from collections import defaultdict
from transformers import BertTokenizer
from functools import partial
from torch.utils.data import DataLoader
from torchmultimodal.models.flava.model import flava_model_for_classification
from datasets import load_dataset
# Including Accelerator
from accelerate import Accelerator

# Create accelerator and define the device
accelerator = Accelerator()
device = accelerator.device

with open("data/vocabs/answers_textvqa_more_than_1.txt") as f:
  vocab = f.readlines()

answer_to_idx = {}
for idx, entry in enumerate(vocab):
  answer_to_idx[entry.strip("\n")] = idx
print(len(vocab))
print(vocab[:5])

dataset = load_dataset("textvqa")

BATCH_SIZE = 2
MAX_STEPS = 3
device = "cuda"

def transform(tokenizer, input):
    batch = {}
    image_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224])])
    image = image_transform(input["image"][0].convert("RGB"))
    batch["image"] = [image]
    
    tokenized=tokenizer(input["question"],return_tensors='pt',padding="max_length",max_length=512)
    batch.update(tokenized)

    ans_to_count = defaultdict(int)
    for ans in input["answers"][0]:
        ans_to_count[ans] += 1
    max_value = max(ans_to_count, key=ans_to_count.get)
    ans_idx = answer_to_idx.get(max_value,0)
    batch["answers"] = torch.as_tensor([ans_idx])
    return batch

tokenizer=BertTokenizer.from_pretrained("bert-base-uncased",padding="max_length",max_length=512)
transform=partial(transform,tokenizer)
dataset.set_transform(transform)

model = flava_model_for_classification(num_classes=len(vocab))
model.to(device)

train_dataloader = DataLoader(dataset["train"], batch_size= BATCH_SIZE)
optimizer = optim.AdamW(model.parameters())
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# Shard Model, gradient, optimizer state, data to accelerators
model, optimizer, train_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, scheduler
)

def main():
    epochs = 1
    for _ in range(epochs):
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            text_inputs = batch["input_ids"]
            img_inputs = batch["image"]
            targets = batch["answers"]
            outputs = model(text = text_inputs, image = img_inputs, labels = targets)
            loss = output.loss
            # accelerator.backward() replace loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            
        scheduler.step()
        print(f"Loss at step {idx} = {loss}")
        if idx >= MAX_STEPS-1:
            break

if __name__ == "__main__":
    main()