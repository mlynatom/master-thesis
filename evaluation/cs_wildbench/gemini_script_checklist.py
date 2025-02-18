from google import genai
from tqdm import tqdm
from datasets import load_from_disk
import time

#load prompt template
with open("evaluation/cs_wildbench/checklist_prompt.md", "r") as f:
    checklist_generation_prompt_template = f.read()

#load data
from datasets import load_from_disk

dataset = load_from_disk("evaluation/cs_wildbench/wildbench_cs")

histories = []
last_queries = []

for conversation in dataset["test"]["conversation_input"]:
    history = ""
    for turn in conversation[:-1]:
        if turn["role"] == "user":
            history += "USER: " + turn["content"] + "\n\n"
        elif turn["role"] == "assistant":
            history += "ASSISTANT: " + turn["content"] + "\n\n"
        

    histories.append(history)
    last_queries.append(conversation[-1]["content"])

prompts = [checklist_generation_prompt_template.format(history=history, user_query=last_query) for history, last_query in zip(histories, last_queries)]

client = genai.Client(api_key="")

# Create the model
generation_config = {
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 1000,
  "response_mime_type": "text/plain",
}

responses = []


for i, prompt in tqdm(enumerate(prompts)):
    if i % 10 == 0:
        time.sleep(62)

    # try generate content until it works (no server error, each prompt can fail multiple times)
    while True:
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-thinking-exp-01-21",
                config=generation_config,
                contents=[prompt],
            )
            break
        except genai.errors.ServerError as e:
            time.sleep(62)
            continue
        except Exception as e:
            raise e
    

    responses.append(response.text)

import json
with open("evaluation/cs_wildbench/gemini_responses_checklist.json", "w") as f:
    json.dump(responses, f)
