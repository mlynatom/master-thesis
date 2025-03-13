from datasets import load_dataset
import time

#logging
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load the dataset
logging.info("Loading dataset")

# this dataset has already fixed encoding using ftfy (as is used by me in the preprocessing steps of other datasets)
dataset = load_dataset("HuggingFaceFW/fineweb-2", "ces_Latn", split="train")
dataset = dataset.remove_columns(["id", "dump", "url", "date", "file_path", "language", "language_score", "language_script", "minhash_cluster_size", "top_langs"])
dataset = dataset.shuffle(seed=42)
dataset = dataset.take(5000000)

logging.info("Dataset loaded")

# Load the old tokenizer
logging.info("Loading old tokenizer")

from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
logging.info("Old tokenizer loaded")


# Train the new tokenizer
logging.info("Training new tokenizer")
start_time = time.time()

def serve_texts():
    for i, example in enumerate(dataset):
        if i % 1000 == 0 and i != 0:
            logging.info(f"Processed {i} examples, elapsed time: {time.localtime(time.time() - start_time)}, estimated time remaining: {time.localtime((time.time() - start_time) / i * len(dataset) - time.time())}")
        yield example["text"]
tokenizer = old_tokenizer.train_new_from_iterator(text_iterator=serve_texts(), vocab_size=25000, length=len(dataset))
logging.info("New tokenizer trained")

#tokenizer merging
old_vocab = set(old_tokenizer.get_vocab())
new_vocab = set(tokenizer.get_vocab())


diff_vocab = new_vocab - old_vocab
new_tokens = list(diff_vocab)

#add the difference between vocabularies (creating union of the two vocabularies)
num_added_tokens = old_tokenizer.add_tokens(new_tokens)
print(num_added_tokens)

old_tokenizer.save_pretrained("tokenizers/llama-3.1-8B-cs_expand_5M")