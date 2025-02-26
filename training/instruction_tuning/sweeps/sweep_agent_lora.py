from unsloth import FastLanguageModel, is_bfloat16_supported

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from datasets import load_from_disk
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from peft import LoftQConfig
import torch

import os

# random state
SEED = 42

# reproducibility
## torch
torch.manual_seed(SEED)

## python
import random
random.seed(SEED)

## numpy
import numpy as np
np.random.seed(SEED)

import wandb

os.environ["WANDB_PROJECT"]="it-lora_params-sweep"


def sweep_train(config=None):

    wandb.init(config=config)
    #set sweep config
    config = wandb.config

    loftq_config = LoftQConfig(loftq_bits=4, loftq_iter=config.loftq_t)
    load_in_4bit = False


    #Load Model
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    #load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model_id = "meta-llama/Llama-3.2-1B"

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
        r = config.r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
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
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )

    # def formatting_prompts_func(examples):
    #     convos = examples["conversations"]
    #     texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    #     return { "text" : texts, }
    
    #TODO load data
    dataset = load_from_disk("data/it/init_mix_cs-en")




    RUN_NAME = f"it_{model_id.split('/')[-1]}-r{config.r}"
    #init trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["validation"],
        dataset_text_field = "text",
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,

        args = TrainingArguments(
            per_device_train_batch_size = 128, #TODO,
            gradient_accumulation_steps = 4, #TODO,
            warmup_ratio = 0.05,
            num_train_epochs = 1, # Set this for 1 full training run.
            learning_rate = 8e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = SEED,
            output_dir = "output_models",
            report_to = "wandb", # Use this for WandB etc
            run_name=RUN_NAME,
            eval_strategy = "epoch",
            #eval_steps = args.eval_steps,
        ),
    )

    #training only on responses (mask out instructions)
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
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


    # #local save model
    # model.save_pretrained(f"models/{RUN_NAME}")
    # tokenizer.save_pretrained(f"models/{RUN_NAME}")

if __name__ == "__main__":
    SWEEP_ID = "gtyrcdyl"
    wandb.agent(SWEEP_ID, sweep_train)