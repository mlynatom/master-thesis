import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from tqdm import tqdm
from datasets import load_from_disk

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

endpoint = ""
model_name = "DeepSeek-R1"
token = ""

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

deepseek_responses = []

for i, prompt in tqdm(enumerate(prompts)):
    try:
        response = client.complete(
            messages=[
                UserMessage(content=prompt),
            ],
            max_tokens=1000,
            model=model_name
        )
        response = response.choices[0].message.content
    except HttpResponseError as e:
        response = None
        

    deepseek_responses.append(response)

import json
with open("evaluation/cs_wildbench/deepseek_responses_checklist.json", "w") as f:
    json.dump(deepseek_responses, f)