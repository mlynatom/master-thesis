from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset
from typing import Dict, List

from tqdm import tqdm

class TokenizerEvaluator:
    def __init__(self, fertility_dataset: str = "BUT-FIT/BUT-LCC", fertility_split: str="test", parity_dataset:str = ""):
        # load evaluation datasets
        ## fertility dataset
        self.fertility_dataset = load_dataset(fertility_dataset, split=fertility_split, streaming=True)

        ##prepare the number of words in the fertility dataset
        def process_fn(examples):
            examples["num_words"] = [len(text.split()) for text in examples["text"]]
            return examples
        
        self.fertility_dataset = self.fertility_dataset.map(process_fn, batched=True, remove_columns=["title", "part"])

        ## parity dataset
        #self.parity_dataset = load_dataset(parity_dataset, split="train")

    def evaluate(self, model_ids: List[str], verbose: bool = False) -> Dict[str, Dict[str, float]]:
        results = {}
        for model_id in model_ids:
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            if verbose:
                print(f"Evaluating tokenizer {model_id}")

            model_results = {
                "fertility": self.evaluate_fertility(tokenizer),
                "parity": self.evaluate_parity(tokenizer)
            }

            results[model_id] = model_results

        return results

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
    def evaluate_parity(self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> float:
        return 0.0

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
            examples["num_tokens"] = [len(tokenizer.tokenize(text)) for text in texts]

            return examples

        #number of words in the fertility dataset - using whitespace splitting - streaming support
        data = self.fertility_dataset.map(process_fn, batched=True, remove_columns=["text"])

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