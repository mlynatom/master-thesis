from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
import tiktoken

from datasets import IterableDataset, load_dataset, load_from_disk


class TokenizerEvaluator:
    def __init__(self, fertility_dataset: str = "BUT-FIT/BUT-LCC", fertility_subset:Optional[str] = None, fertility_split: str = "test", parity_dataset: str = "/mnt/data/factcheck/czeng20/hf_dataset", parity_split: str = "test"):
        # load evaluation datasets
        # fertility dataset
        if fertility_subset is not None:
            self.fertility_dataset = load_dataset(
                fertility_dataset, fertility_subset, split=fertility_split, streaming=True)
        else:
            self.fertility_dataset = load_dataset(
                fertility_dataset, split=fertility_split, streaming=True)

        # prepare the number of words in the fertility dataset
        def process_fn(examples):
            examples["num_words"] = [len(text.split())
                                     for text in examples["text"]]
            return examples

        if fertility_dataset == "BUT-FIT/BUT-LCC":
            self.fertility_dataset = self.fertility_dataset.map(
                process_fn, batched=True, remove_columns=["title", "part"])
        
        elif fertility_subset == "HuggingFaceFW/fineweb-2":
            self.fertility_dataset = self.fertility_dataset.map(
                process_fn, batched=True, remove_columns=["id", "dump", "url", "date", "file_path", "language", "language_score", "language_script", "minhash_cluster_size", "top_langs"])
        else:
            self.fertility_dataset = self.fertility_dataset.map(
                process_fn, batched=True)

        # parity dataset
        self.parity_dataset = load_from_disk(parity_dataset)[parity_split]

    def evaluate(self, model_ids: List[str], verbose: bool = False) -> Dict[str, Dict[str, float]]:
        # todo preprocess model_ids to groups based on the tokenizer identity
        unique_tokenizer_ids = self.get_unique_tokenizer_ids(model_ids)
        if verbose:
            print(f"Found {len(unique_tokenizer_ids)} unique tokenizers")

        results = {}
        for model_id, all_models in unique_tokenizer_ids.items():
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            except:
                try:
                    tokenizer = tiktoken.encoding_for_model(model_id)
                except:
                    raise ValueError("Tokenizer not found")

            if verbose:
                print(f"Evaluating tokenizer {model_id}")

            model_results = {
                "fertility": self.evaluate_fertility(tokenizer),
                "parity": self.evaluate_parity(tokenizer),
                "all_models": all_models
            }

            results[model_id] = model_results

        return results

    def get_unique_tokenizer_ids(self, model_ids: List[str]) -> Dict[str, List[str]]:
        unique_tokenizer_ids = {}
        for model_id in model_ids:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            except:
                unique_tokenizer_ids[model_id] = [model_id]
                continue


            found_identical_tokenizer = False
            for unique_id in unique_tokenizer_ids.keys():
                unique_tokenizer = AutoTokenizer.from_pretrained(unique_id)

                # compare vocabulary sizes (fast) and vocabularies (slower)
                if tokenizer.vocab_size == unique_tokenizer.vocab_size and tokenizer.get_vocab() == unique_tokenizer.get_vocab():
                    unique_tokenizer_ids[unique_id].append(model_id)
                    found_identical_tokenizer = True
                    break

            if not found_identical_tokenizer:
                unique_tokenizer_ids[model_id] = [model_id]

        return unique_tokenizer_ids

    """
    (Petrov et al., 2024)
    To demonstrate that the above examples are not anecdotal evidence, we introduce the notion
    of tokenizer parity to systematically assess how fairly tokenizers treat equivalent sentences
    in different languages. Parity occurs when a tokenizer exhibits similar tokenized lengths
    for the same sentence in different languages. Take a sentence sA in language A and its
    translation sB to language B. Then, a tokenizer t achieves parity for A with respect to B
    at sA and sB if |t(sA)|/|t(sB)| ≈ 1, where t(sA) is the tokenization of the sentence sA and
    |t(sA)| represents its length. The ratio |t(sA)|/|t(sB)| is the premium for A relative to B.
    """

    def evaluate_parity(self, tokenizer) -> float:
        
        def process_fn(examples):
            cs_texts = examples["text_cs"]
            en_texts = examples["text_en"]
            if isinstance(tokenizer, PreTrainedTokenizerFast) or isinstance(tokenizer, PreTrainedTokenizer):
                examples["num_tokens_cs"] = [
                    len(tokenizer.tokenize(text)) for text in cs_texts]
                examples["num_tokens_en"] = [
                    len(tokenizer.tokenize(text)) for text in en_texts]
            else:
                examples["num_tokens_cs"] = [
                    len(tokenizer.encode(text)) for text in cs_texts]
                examples["num_tokens_en"] = [
                    len(tokenizer.encode(text)) for text in en_texts]

            return examples

        data = self.parity_dataset.map(
            process_fn, batched=True, remove_columns=["text_cs", "text_en"])

        parity = 0

        # if the dataset is streamed (IterableDataset)
        if isinstance(data, IterableDataset):
            for i, dat in tqdm(enumerate(data)):
                num_tokens_cs = dat["num_tokens_cs"]
                num_tokens_en = dat["num_tokens_en"]

                current_parity = num_tokens_cs / num_tokens_en

                # compute running average of parity
                parity = parity * i / (i + 1) + current_parity / (i + 1)

        else:
            num_tokens_cs = np.array(data["num_tokens_cs"])
            num_tokens_en = np.array(data["num_tokens_en"])

            parity = float(np.average(num_tokens_cs / num_tokens_en))

        return parity

    """
    (Ali et al., 2024)
    Fertility, the most common metric to evaluate a
    tokenizer’s performance (Scao et al., 2022; Stollenwerk, 2023; Rust et al., 2021), is defined as the
    average number of tokens that are required to represent a word or document. For a tokenizer T and
    dataset A, the fertility can be calculated as the number of tokens in A (when T is applied) divided by
    the number of words in A. We calculate the fertility on a held-out set (10,000 documents), which was
    not used for the tokenizer training. For calculating the words of a document, we used whitespace splitting. 
    Higher fertility scores correspond to weaker compression capabilities of the tokenizer
    """

    def evaluate_fertility(self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> float:
        def process_fn(examples):
            texts = examples["text"]
            if isinstance(tokenizer, PreTrainedTokenizerFast) or isinstance(tokenizer, PreTrainedTokenizer):
                examples["num_tokens"] = [
                    len(tokenizer.tokenize(text)) for text in texts]
            else:
                examples["num_tokens"] = [len(tokenizer.encode(text)) for text in texts]

            return examples

        # number of words in the fertility dataset - using whitespace splitting - streaming support
        data = self.fertility_dataset.map(
            process_fn, batched=True, remove_columns=["text"])

        num_tokens = 0
        num_words = 0
        for dat in tqdm(data):
            num_tokens += dat["num_tokens"]
            num_words += dat["num_words"]

        return num_tokens / num_words


if __name__ == '__main__':
    print("This is a tokenizer evaluator")

    model_ids = ["meta-llama/Llama-3.2-3B"]

    tokenizer_evaluator = TokenizerEvaluator(model_ids)

    results = tokenizer_evaluator.evaluate(verbose=True)
