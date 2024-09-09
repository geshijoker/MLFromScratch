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

save_directory = 'fine_tuned_model'

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

optimizer = AdamW(model.parameters(), lr=3e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

def main():
    accelerator = Accelerator(gradient_accumulation_steps=2, mixed_precision="bf16")
    device = accelerator.device
    
    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )
    
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    
    model.train()
    for epoch in range(num_epochs):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            with accelerator.autocast():
                loss = outputs.loss
                accelerator.backward(loss)
    
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if accelerator.is_local_main_process:
                print("Accelerate is the best")
            progress_bar.update(1)
    
    accelerator.wait_for_everyone()
    accelerator.save_model(model, save_directory)
    
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(
    #     save_directory,
    #     is_main_process=accelerator.is_main_process,
    #     save_function=accelerator.save,
    # )
    # model = AutoModel.from_pretrained(save_directory)
    
    unwrapped_model = accelerator.unwrap_model(model)
    path_to_checkpoint = os.path.join(save_directory, "pytorch_model.bin")
    unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))
    
    metric= evaluate.load('glue', 'mrpc')
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
    
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'])
    
    results = metric.compute()
    print(results)

if __name__ == "__main__":
    main()
