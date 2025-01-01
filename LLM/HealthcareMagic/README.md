# Supervised finetuning of Llama-3 on Private Dataset

<!-- ABOUT THE PROJECT -->
## About The Project

This project guide practitioners in supervised finetuning (SFT) a large language models (LLM) on a private dataset. The llama-factory framework is used which built on the huggingface transformers, peft, trainers libraries to conduct the experiments. A question and answer dataset of chats betweeen patient and doctor on health care related topics is used to finetune on Llama-3-8B models. Various configures of full, lora, and quantized lora finetuning are explored.

| Recipe | Resource |
| --- | --- |
| code base | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| library | [huggingface parameter efficient tuning (peft)](https://huggingface.co/docs/peft/en/package_reference/lora)|
| dataset | [ChatDoctor HealthcareMagic dataset](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) |
| base model | [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| Monitor | [WanDB](https://wandb.ai/site/) |

The selected finetuned models are located at [Huggingface Collection of Models](https://huggingface.co/collections/geshijoker/chatdoctor-67704eef2b1891292b89242d)

<!-- STEPS -->
## Steps

1. Get a free API Key at [Huggingface user access tokens](https://huggingface.co/docs/hub/en/security-tokens) and [WanDB user access tokens](https://wandb.ai/authorize)
2. Install pre-required packages
   ```sh
   pip install tensorrt
   pip install pillow
   pip install tf-keras
   pip install autoawq
   pip install deepspeed==0.15.4
   ```
3. On the linux machine with GPUs, log in huggingface with the token
   ```sh
   huggingface-cli login
   ```
4. Go to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) github page and follow the "Getting Started" instructions.
   ```sh
   git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   pip install -e ".[torch,metrics]"
   pip install --no-deps -e .
   ```
5. Add private dataset copy the following content to "./data/dataset_info.json"
   ```python
   "healthcaremagic": {
    "hf_hub_url": "lavita/ChatDoctor-HealthCareMagic-100k"
   },
   ```
6. Make changes to the *.ymal configure files and the corresponding directory
   ```sh
   cp llama3_full_sft_healthcare.yaml exampels/train_full/
   cp llama3_lora_sft_healthcare.yaml exampels/train_lora/
   cp llama3_qlora_sft_awq_healthcare.yaml exampels/train_qlora/
   ```
7. Finetune the model
   ```sh
   llamafactory-cli train examples/train_lora/llama3_lora_sft_healthcare.yaml
   ```

I first took 1% of all data to conduct pre-experiments to study the general setups such as learing rate, hyperparameters (target_modules, rank, alpha) of lora, dataset (private, identity, alpaca) combinations to filter out bad practices. Then perform experiments on the entire dataset using 8 A100 80G GPUs to output checkpoints. It's a lot of fun and here are the takeaways.

<!-- Takeaways -->
## Takeaways of Resource Consumption

### Finetuning Types
Here the rank for lora and qlora is 64, and all linear layers are target modules.
With the same batch size, if the running time of lora is 1, full finetuning takes about 2.22, qlora takes about 1.56.
With the same batch size, if the peak memory of full finetuning is 1, lora takes about 0.49, qlora takes about 0.28.

### Target Modules
Comparing finetuning all linear layers (lora_all) and only (q_proj, v_proj) (lora_qv), if the time of of lora_qv is 1, lora_all takes about 1.38. The memory difference is very small, lora_all takes only 1% more than lora_qv. The same for qlora.

### Rank
In the setting, I always set alpha as 2 times of rank as convention. I tested a few configurations, here I use rank=8 and rank=128 as examples. The time difference is almost 0, lora_128 takes 1% more memory than lora_8.

## Takeaways of Performance
### Adding Other Instruction Data
I tested adding extra alpaca and identity dataset as training data, the evaluation loss is higher than a pure private domain dataset but converges as fast.

### Finetuning Types
Qlora adds a small loss than lora but unstable at the beginning.

### Target Modules
Only finetune q,v shows a much higher loss than finetune all linear layers

### Rank
Higher rank shows similar loss at the beginning of trainning with loswer rank but converges much faster.

### Overall Performance
Adam optimizer and Cosine scheduler with 5% warm-up is used. The following table presents the evaluation results (entire dataset) for different models and fine-tuning strategies on several metrics:

| Name                       | Checkpoint                   | Rouge1 | RougeL | Meteor | Bert Score |
|----------------------------|------------------------------|--------|--------|--------|------------|
| Baseline Instruct Model    | Meta-Llama-3-8B-Instruct     | 0.254  | 0.128  | 0.222  | 0.747      |
| Full Fine-tune from Inst   | sft_llama3_instruct_full     | 0.315  | 0.189  | 0.238  | 0.782      |
| LoRA SFT from Inst         | sft_llama3_instruct_lora_all | 0.271  | 0.143  | 0.194  | 0.774      |
| LoRA SFT from Base         | sft_llama3_lora_all          | 0.239  | 0.113  | 0.211  | 0.735      |
| QLoRA SFT from Inst        | sft_llama3_instruct_qlora_all| 0.137  | 0.071  | 0.102  | 0.679      |
| QLoRA SFT from Base        | sft_llama3_qlora_all         | 0.192  | 0.090  | 0.159  | 0.718      |


## Suggestions of Practices
1. Full finetuning is much better than lora when there is a drift in data distribution. It maintains much better natural language abilities although it takes about 2 times of time and memory.
2. Lora is able to inject identity to the model, for example, it would start an answer with "Thank you for posting on Chat Doctor". However, it cannot retain the natural language ability (it generates repeated contents) and inject domain specific knowledge. Lora finetuning all linear layers exibits salient better performance than only fine-tune the q_proj and v_proj. Although it takes more time, it almost doesn't increase memory consumption.
3. qLora is only suggested when the memory is a big concern.
4. Finetune from base model instead of chat model retains better natural language abilities.
5. All methods struggle to inject domain knowledges (full finetuning could be an exception but I have not tested it with domain specific multichoice questions), continuous pretraining with a small portion mixture of pretraining data could be good for it.
6. The final suggestion is full-finetuning is always better in performance although takes much resources. Lora is suggested to use after continuous pretraining and only finetune on a small set of high quality QA data for very few epochs to offer the chatbot an identity or style of languages.

## Trouble Shooting
1. Lora finetuned model tends to generate repeated contents, it can be mitigated by adding repetition penalty. Some resources suggested to lower temperature and increase top-p to mitigate the problem but I didn't see much effects.
2. qlora is very bad in terms of performance, although it can inject identity as well. It messes up the tokens as generated contents may not have spaces between words. qlora training is much less stable and hard to converge.
3. The training loss curve shows sudden jumps at the beginning of each epoch. The potential reasons are, a) the loss is evaluated as epoch average (not applicable), b) the sampler tends to group data with same length together where longer answers may hard to predict. It doesn't affect the convergence but still need to check furthur.
