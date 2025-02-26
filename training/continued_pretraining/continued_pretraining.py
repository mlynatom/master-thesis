from unsloth import FastLanguageModel, is_bfloat16_supported

from unsloth import UnslothTrainingArguments, UnslothTrainer
from datasets import load_dataset

import torch

import os
os.environ["WANDB_PROJECT"]="cp_experiment"

import argparse

# command line arguments
parser = argparse.ArgumentParser(description='Instruction Tuning')

parser.add_argument('--model_id', type=str, help='Model ID')
parser.add_argument('--random_state', type=int, help='Random State (SEED)')
parser.add_argument('--dtype', type=str, default=None, help='Data Type')
parser.add_argument('--load_in_4bit', type=bool, default=True, help='Load in 4bit')
parser.add_argument('--max_seq_length', type=int, default=4096, help='Max Sequence Length')
parser.add_argument('--lora_r', type=int, default=256, help='LoRA rank')
parser.add_argument('--lora_alpha', type=int, default=512, help='LoRA alpha')
parser.add_argument('--lora_dropout', type=float, default=0, help='LoRA dropout')
parser.add_argument('--bias', type=str, default='none', help='Bias')
parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth', help='Use Gradient Checkpointing')
parser.add_argument('--use_rslora', type=bool, default=True, help='Use Rank Stabilized LoRA')
parser.add_argument('--chat_template', type=str, default='llama-3.1', help='Chat Template')
parser.add_argument('--dataset_id', type=str, default='ctu-aic/cs_instruction_tuning_collection', help='Dataset ID')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient Accumulation Steps')
parser.add_argument('--warmup_ratio', type=float, default=0.01, help='Warmup Ratio')
parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of Training Epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--logging_steps', type=int, default=5, help='Logging Steps')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight Decay')
parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Learning Rate Scheduler Type')
parser.add_argument('--eval_strategy', type=str, default='steps', help='Evaluation Strategy')
parser.add_argument('--eval_steps', type=int, default=50, help='Evaluation Steps')
parser.add_argument('--output_dir', type=str, default='output_models', help='Output Directory')

args = parser.parse_args()

# random state
SEED = args.random_state

# reproducibility
## torch
torch.manual_seed(SEED)

## python
import random
random.seed(SEED)

## numpy
import numpy as np
np.random.seed(SEED)


#Load Model
max_seq_length = args.max_seq_length # Choose any! We auto support RoPE Scaling internally!
dtype = args.dtype # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = args.load_in_4bit # Use 4bit quantization to reduce memory usage. Can be False.

model_id = args.model_id

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# Init LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = args.lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "embed_tokens", "lm_head"],
    lora_alpha = args.lora_alpha,
    lora_dropout = args.lora_dropout, # Supports any, but = 0 is optimized
    bias = args.lora_bias,    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = args.use_gradient_checkpointing, # True or "unsloth" for very long context
    random_state = SEED,
    use_rslora = args.use_rslora,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# prepare data
from datasets import load_dataset

# this dataset has already fixed encoding using ftfy (as is used by me in the preprocessing steps of other datasets)
dataset = load_dataset("HuggingFaceFW/fineweb-2", "ces_Latn")
#we need only texts
dataset = dataset.remove_columns(["id", "dump", "url", "date", "file_path", "language", "language_score", "language_script", "minhash_cluster_size", "top_langs"])
#shuffle to be sure we select "random sample"
dataset = dataset.shuffle(seed=42)

def preprocess_function(examples):
    return {"text": [example + tokenizer.eos_token for example in examples["text"]]}

dataset = dataset.map(preprocess_function, batched=True)



RUN_NAME = f"cp_{model_id.split('/')[-1]}-bs{args.batch_size}-lr{args.learning_rate}-e{args.num_train_epochs}-s{SEED}"
#init trainer
trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    #eval_dataset = dataset["validation"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_ratio = args.warmup_ratio,
        #num_train_epochs = args.num_train_epochs, # Set this for 1 full training run.
        max_steps = 1000,
        learning_rate = args.learning_rate,
        embedding_learning_rate = args.learning_rate/5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = args.logging_steps,
        optim = "adamw_8bit",
        weight_decay = args.weight_decay,
        lr_scheduler_type = args.lr_scheduler_type,
        seed = SEED,
        output_dir = args.output_dir,
        report_to = "wandb", # Use this for WandB etc
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