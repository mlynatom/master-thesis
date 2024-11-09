from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset
from typing import Dict, List

from tokens import hf_token


class TokenizerEvaluator:
    def __init__(self, model_ids: List[str], fertility_dataset: str = "BUT-FIT/BUT-LCC", parity_dataset:str = ""):
        self.model_ids = model_ids

        # load evaluation datasets
        ## fertility dataset
        self.fertility_dataset = load_dataset(fertility_dataset, split="train", token=hf_token)

        ## parity dataset
        #self.parity_dataset = load_dataset(parity_dataset, split="train")

    def evaluate(self, verbose: bool = False) -> Dict[str, Dict[str, float]]:
        results = {}
        for model_id in self.model_ids:
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            if verbose:
                print(f"Evaluating tokenizer {model_id}")

            model_results = {
                "fertility": self.evaluate_fertility(tokenizer),
                "parity": self.evaluate_parity(tokenizer)
            }

            results[model_id] = model_results

        return results

            
    def evaluate_parity(self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> float:
        pass

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
        #number of words in the fertility dataset - using whitespace splitting
        num_words = self.fertility_dataset.map(lambda x: len(x.split())).reduce(lambda x, y: x + y)

        #number of tokens in the fertility dataset
        num_tokens = self.fertility_dataset.map(lambda x: len(tokenizer.tokenize(x))).reduce(lambda x, y: x + y)

        return num_tokens / num_words


if __name__ == '__main__':
    print("This is a tokenizer evaluator")

    model_ids = ["meta-llama/Llama-3.2-3B"]

    tokenizer_evaluator = TokenizerEvaluator(model_ids)

    results = tokenizer_evaluator.evaluate(verbose=True)