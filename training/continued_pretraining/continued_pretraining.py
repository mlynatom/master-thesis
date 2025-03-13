from unsloth import FastLanguageModel, is_bfloat16_supported

from unsloth import UnslothTrainingArguments, UnslothTrainer
from datasets import load_dataset
import random
import numpy as np
from peft import LoftQConfig

import torch

import os
os.environ["WANDB_PROJECT"]="unsloth_cp_experiment"


# params
SEED = 42
model_id =  "meta-llama/Llama-3.1-8B"
max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
# Training params
batch_size = 8
gradient_accumulation_steps = 128
warmup_ratio = 0.05
max_steps = 61234
learning_rate = 5e-4
embedding_learning_rate = learning_rate/2
weight_decay = 0.01
# LoRA params
lora_r = 256 # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
lora_alpha = 1

# reproducibility
## torch
torch.manual_seed(SEED)

## python
random.seed(SEED)

## numpy
np.random.seed(SEED)


#Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

#LoftQ config
loftq_config = LoftQConfig(loftq_bits=4)

# Init LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens", "lm_head"],
    lora_alpha = 1,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = SEED,
    use_rslora = True,  # We support rank stabilized LoRA
    loftq_config = loftq_config, # And LoftQ
)

# prepare data
# from datasets import load_dataset

# # this dataset has already fixed encoding using ftfy (as is used by me in the preprocessing steps of other datasets)
# dataset = load_dataset("HuggingFaceFW/fineweb-2", "ces_Latn", split="train", streaming=True)
# #we need only texts
# dataset = dataset.remove_columns(["id", "dump", "url", "date", "file_path", "language", "language_score", "language_script", "minhash_cluster_size", "top_langs"])
# #shuffle to be sure we select "random sample"
# dataset = dataset.shuffle(seed=42)

# def preprocess_function(examples):
#     return {"text": [example + tokenizer.eos_token for example in examples["text"]]}

# dataset = dataset.map(preprocess_function, batched=True)

from datasets import load_from_disk

dataset = load_from_disk("data/pretraining/fineweb-2_ces_Latn_19531250_llama_preprocessed")
dataset = dataset.select(range(10000))
#dataset = dataset.to_iterable_dataset()
dataset



RUN_NAME = f"cp_{model_id.split('/')[-1]}-cs"
#init trainer
trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 12,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_ratio = warmup_ratio,
        num_train_epochs = 1, # Set this for 1 full training run.
        #max_steps = max_steps,
        learning_rate = learning_rate,
        embedding_learning_rate = embedding_learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = weight_decay,
        lr_scheduler_type = "cosine",
        seed = SEED,
        output_dir = f"models/cp_{RUN_NAME}",
        report_to = "none", # Use this for WandB etc
        run_name=RUN_NAME,
        # eval_strategy = args.eval_strategy,
        # eval_steps = args.eval_steps,
    ),
)

#Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

#training
trainer_stats = trainer.train()

#Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


#local save model
model.save_pretrained(f"models/{RUN_NAME}")
tokenizer.save_pretrained(f"models/{RUN_NAME}")