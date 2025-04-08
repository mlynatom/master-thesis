# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Perplexity Metric."""
from unsloth import FastLanguageModel
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import argparse
import os

class Perplexity:
    def __init__(self, model_id:str, device:str=None, load_in_4bit:bool=True)->None:
        self.model_id = model_id

        #check device
        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        
        # #load model and move to desired device
        # if load_in_16bit:
        #     self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16)
        # else:
        #     self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        print(load_in_4bit, "load_in_4bit")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_id, # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length = 1024,
            dtype = torch.bfloat16,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)


        #self.model.to(device)


        # #load tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def compute(self, predictions, batch_size: int = 16, add_start_token: bool = True, max_length=None):
        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm(range(0, len(predictions), batch_size)):
            end_index = min(start_index + batch_size, len(predictions))

            #compute encodings
            encodings = self.tokenizer(
                predictions[start_index:end_index],
                add_special_tokens=False,
                padding=True,
                truncation=True if max_tokenized_len else False,
                max_length=max_tokenized_len,
                return_tensors="pt",
                return_attention_mask=True,
            )

            encoded_batch = encodings["input_ids"]
            attn_mask = encodings["attention_mask"]


            # check that each input is long enough:
            if add_start_token:
                assert torch.all(torch.ge(attn_mask.sum(1), 1)), "Each input text must be at least one token long."
            else:
                assert torch.all(
                    torch.ge(attn_mask.sum(1), 2)
                ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0))
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat([torch.ones(bos_tokens_tensor.size(), dtype=torch.int64), attn_mask], dim=1)


            #now move to gpu
            encoded_batch = encoded_batch.to(self.device)
            attn_mask = attn_mask.to(self.device)

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
    

if __name__ == "__main__":
    #parse arguments - model_id, device, load_in_16bit, dataset_id, batch_size, add_start_token, max_length
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_id", type=str, required=True, help="Model ID to compute perplexity.")
    argparser.add_argument("--device", type=str, default=None, help="Device to run the model on. Default is 'cuda' if available.")
    argparser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, help="Whether to load the model in 4-bit precision.")
    argparser.add_argument("--batch_size", type=int, default=16, help="Batch size to use for computing perplexity.")
    argparser.add_argument("--add_start_token", action=argparse.BooleanOptionalAction, help="Whether to add <BOS> token to the input.")
    argparser.add_argument("--max_length", type=int, default=None, help="Maximum length of the input sequence.")
    argparser.add_argument("--dataset_id", type=str, default="HuggingFaceFW/fineweb-2", help="Dataset ID to use for computing perplexity.")
    argparser.add_argument("--split", type=str, default="test", help="Split of the dataset to use for computing perplexity.")
    argparser.add_argument("--subset", type=str, default=None, help="Subset of the dataset to use for computing perplexity.")
    argparser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use from the dataset.")
    argparser.add_argument("--output_dir", type=str, default=None, help="Output directory to save the results.")

    args = argparser.parse_args()

    #check if dataset_id is directory or Hugging Face hub id
    if os.path.exists(args.dataset_id):
        dataset = load_from_disk(args.dataset_id)
        #select the split
        dataset = dataset[args.split]
        if args.subset is not None:
            raise ValueError("Subset is not supported for datasets loaded from disk.")
        
        
    else:
        #check if dataset_id is a valid Hugging Face dataset
        try:
            dataset = load_dataset(args.dataset_id, args.subset, split=args.split)
        except ValueError:
            print(f"Dataset {args.dataset_id} not found. Please check the dataset ID.")
            exit(1)

    print(args.load_in_4bit, "load_in_4bit")
    perplexity_evaluator = Perplexity(model_id=args.model_id, device=args.device, load_in_4bit=False if args.load_in_4bit is None else True)
    if args.num_samples is None:
        result = perplexity_evaluator.compute(dataset["text"], args.batch_size, args.add_start_token, args.max_length)
    else:
        result = perplexity_evaluator.compute(dataset["text"][:args.num_samples], args.batch_size, args.add_start_token, args.max_length)

    print(result)

    model_name = args.model_id.split("/")[-1]
    if model_name == "final":
        model_name = args.model_id.split("/")[-2]

    dataset_name = args.dataset_id.split("/")[-1]
    

    with open(os.path.join(args.output_dir, f"perplexity_results_{model_name}_{dataset_name}.txt"), "w") as f:
        f.write(str(result))

    