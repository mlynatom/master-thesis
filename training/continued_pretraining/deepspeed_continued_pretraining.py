from datasets import load_dataset
import random
import numpy as np
from peft import LoftQConfig, get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments
import torch
import argparse
import deepspeed

import os
os.environ["WANDB_PROJECT"]="deepspeed_cp_experiment"


def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr = 5e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = \
    {
        "non_embeddings" : {},
        "embeddings"     : {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[:-len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".")+1:]
            print(f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}.")
            param_groups["embeddings"]    [name] = param
        else:
            param_groups["non_embeddings"][name] = param
        pass
    pass

    optimizer_grouped_parameters = [
        {
            "params"       : list(param_groups["non_embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : lr,
        },
        {
            "params"       : list(param_groups["embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer

@dataclass
class UnslothTrainingArguments(TrainingArguments):
    embedding_learning_rate : Optional[float] = field(
        default = None,
        metadata = {"help" : "Different learning rates for embeddings and lm_head."}
    )

class UnslothTrainer(SFTTrainer):
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None: return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        
        return self.optimizer

def main(args):
    # PARAMS
    SEED = 42
    model_id =  args.model_name_or_path #"meta-llama/Llama-3.1-8B"
    max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
    # Training params
    batch_size = args.per_device_train_batch_size # 8
    gradient_accumulation_steps = args.gradient_accumulation_steps #128
    warmup_ratio = 0.05
    max_steps = args.max_train_samples
    learning_rate = args.lr # 5e-4
    embedding_learning_rate = learning_rate/2
    weight_decay = args.weight_decay # 0.01
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

    #load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    #load model
    #TODO load in 4 bit support?
    #float16 set by default (better for gradient accumulation according to deepspeed)
    #TODO max_seq_length not here
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    #LoftQ config
    loftq_config = LoftQConfig(loftq_bits=4)

    #init LoRA
    #TODO checkpointing not here
    lora_config = LoraConfig(r=lora_r, 
                             alpha=lora_alpha, 
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj",
                                             "embed_tokens", "lm_head"],
                             lora_dropout=0,
                             bias="none",
                             use_rslora=True,
                             init_lora_weights="loftq",
                             loftq_config=loftq_config)


    model = get_peft_model(model, lora_config)

    # prepare data
    # this dataset has already fixed encoding using ftfy (as is used by me in the preprocessing steps of other datasets)
    dataset = load_dataset("HuggingFaceFW/fineweb-2", "ces_Latn", split="train", streaming=True)
    #we need only texts
    dataset = dataset.remove_columns(["id", "dump", "url", "date", "file_path", "language", "language_score", "language_script", "minhash_cluster_size", "top_langs"])
    #shuffle to be sure we select "random sample"
    dataset = dataset.shuffle(seed=42)

    def preprocess_function(examples):
        return {"text": [example + tokenizer.eos_token for example in examples["text"]]}

    dataset = dataset.map(preprocess_function, batched=True)


    RUN_NAME = f"deepspeed_{model_id.split('/')[-1]}-cs"
    #init trainer
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,

        args = UnslothTrainingArguments(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_ratio = warmup_ratio,
            #num_train_epochs = args.num_train_epochs, # Set this for 1 full training run.
            max_steps = max_steps,
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
            report_to = "wandb", # Use this for WandB etc
            run_name=RUN_NAME,
            gradient_chekpointing = True,
            deepspeed = "deepspeed_config.json",
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    main(args)