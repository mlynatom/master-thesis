from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, load_from_disk
import random
import numpy as np
from peft import LoftQConfig

import torch

import os
#os.environ["WANDB_PROJECT"]="unsloth_cp_experiment"
os.environ["NEPTUNE_PROJECT"] = "mlynatom/thesis"


# params
SEED = 42
#model_id =  "/home/mlynatom/master-thesis-repository-tomas-mlynar/models/Llama-3.1-8B-cs_expand_5M_subword_resizing"
model_id =  "meta-llama/Llama-3.1-8B-Instruct"
#model_id =  "meta-llama/Llama-3.1-8B"
#model_id = "/mnt/personal/mlynatom/thesis_models/cp_Llama-3.1-8B-cs_expand_5M_subword_resizing-full_fineweb-2_seed42_samples500000/merge_16bit"
#model_id = "/mnt/personal/mlynatom/thesis_models/cp_Llama-3.1-8B-cs_expand_5M_subword_resizing-full_cs_fineweb2-cs_finewebedu-en_31_500k_seed42_samples500000/merge_16bit"
#model_id = "/mnt/personal/mlynatom/thesis_models/cp_Llama-3.1-8B-full_cs_fineweb2_seed42_neptune_bs128_samples500000/merge_16bit"
#model_id = "/mnt/personal/mlynatom/thesis_models/cp_Llama-3.1-8B-full_fineweb2-cs_finewebedu-en_31_500k_seed42_samples500000/merge_16bit"
#model_id = "/mnt/personal/mlynatom/thesis_models/cp_Llama-3.1-8B-full_cs_fineweb2_seed42_neptune_bs128_samples500000/merge_16bit_v2"
#model_id = "/mnt/personal/mlynatom/thesis_models/cp_Llama-3.1-8B-full_fineweb2-cs_finewebedu-en_31_500k_seed42_samples500000/merge_16bit_v2"

max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.bfloat16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
# Training params
batch_size = 128
gradient_accumulation_steps = 8
warmup_ratio = 0.05
learning_rate = 8e-4
weight_decay = 0.001
# LoRA params
lora_r = 16 # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", #"unsloth", # True or "unsloth" for very long context
    random_state = SEED,
    use_rslora = False,  # We support rank stabilized LoRA
    init_lora_weights ='loftq',
    loftq_config = loftq_config, # And LoftQ
)

print("Model loaded")

# prepare data
#dataset_id = "ctu-aic/cs_instruction_tuning_collection"
#dataset_id = "/mnt/personal/mlynatom/data/it/mix_11_cs_en"
dataset_id = "/mnt/personal/mlynatom/data/it/mix_11_cs_en_alpaca_dolly"
dataset_name = dataset_id.split("/")[-1]

# from datasets import load_dataset 

# this dataset has already fixed encoding using ftfy (as is used by me in the preprocessing steps of other datasets)
#dataset = load_dataset(dataset_id)
dataset = load_from_disk(dataset_id)

dataset = dataset.shuffle(seed=SEED)
#preprocess function
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

print("dataset processing done")

splitted_id = model_id.split('/')
if splitted_id[-1] == "merge_16bit":
    model_name = splitted_id[-2]
elif splitted_id[-1] == "merge_16bit_v2":
    model_name = splitted_id[-2] + "_v2"
else:
    model_name = splitted_id[-1]

RUN_NAME = f"it-{model_name}-{dataset_name}"
#init trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    #data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_ratio = warmup_ratio,
        num_train_epochs = 1, # Set this for 1 full training run.
        #max_steps = max_steps,
        learning_rate = learning_rate,
        fp16 = False,
        bf16 = True,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = weight_decay,
        lr_scheduler_type = "linear",
        seed = SEED,
        output_dir = f"/mnt/personal/mlynatom/thesis_models/{RUN_NAME}",
        report_to = "neptune", # Use this for WandB etc
        run_name=RUN_NAME,
        eval_strategy = "steps",
        eval_steps = 20,
        save_strategy = "steps",
        save_steps = 5,
        save_total_limit = 2,
    ),
)

print("Trainer initialized")

# trainer = train_on_responses_only(
#     trainer,
#     instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
#     response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
# )

# print("Trainer set to train on responses only")

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
model.save_pretrained(f"/mnt/personal/mlynatom/thesis_models/{RUN_NAME}/final")
tokenizer.save_pretrained(f"/mnt/personal/mlynatom/thesis_models/{RUN_NAME}/final")