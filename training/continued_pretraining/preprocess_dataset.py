from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_fast=True)

# # this dataset has already fixed encoding using ftfy (as is used by me in the preprocessing steps of other datasets)
dataset_cs = load_dataset("HuggingFaceFW/fineweb-2", "ces_Latn", split="train")
#we need only texts
dataset_cs = dataset_cs.remove_columns(["id", "dump", "url", "date", "file_path", "language", "language_score", "language_script", "minhash_cluster_size", "top_langs"])
#shuffle to be sure we select "random sample"
dataset_cs = dataset_cs.shuffle(seed=42)

dataset = dataset_cs.select(range(19531250))

def preprocess_function(examples):
    return {"text": [example + tokenizer.eos_token for example in examples["text"]]}

dataset = dataset.map(preprocess_function, batched=True, num_proc=8)

dataset.save_to_disk("data/pretraining/fineweb-2_ces_Latn_19531250_llama_preprocessed")