import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm
import evaluate

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler

from accelerate import Accelerator
import os
import deepspeed
import torch.distributed as dist
from safetensors.torch import load_file

if not dist.is_initialized():
    dist.init_process_group(backend='nccl')  # or 'gloo' depending on your setup

save_directory = 'deepspeed_fine_tuned_model'

accelerator = Accelerator()
device = accelerator.device

raw_datasets = load_dataset('glue', 'mrpc')
checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')
print(tokenized_datasets['train'].column_names)

train_dataloader = DataLoader(
    tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets['validation'], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break
print({k: v.shape for k, v in batch.items()})

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

flag = True
if save_directory:
    path_to_checkpoint = os.path.join(save_directory, "model.safetensors")
    if os.path.isfile(path_to_checkpoint):
        print('file exists')
        state_dict = load_file(path_to_checkpoint)
        model.load_state_dict(state_dict)
        flag = False

optimizer = AdamW(model.parameters(), lr=3e-5)

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

if flag:
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
    
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
    accelerator.wait_for_everyone()
    if save_directory:
        accelerator.wait_for_everyone()
        accelerator.save_model(model, save_directory)

# Function to gather tensors from all GPUs
def gather_tensors(tensor):
    tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)

metric= evaluate.load('glue', 'mrpc')
model.eval()
for batch in eval_dataloader:
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # Gather predictions and references from all GPUs
    gathered_predictions = gather_tensors(predictions)
    gathered_references = gather_tensors(batch['labels'])
    
    # Add gathered predictions and references to the metric
    # if dist.get_rank() == 0:  # Only compute on the main process
    if accelerator.is_main_process:
        metric.add_batch(predictions=gathered_predictions, references=gathered_references)

if accelerator.is_main_process:
    results = metric.compute()
    print(results)