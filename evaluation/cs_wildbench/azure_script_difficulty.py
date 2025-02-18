import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from tqdm import tqdm
import json

prompt_template = """#Instrukce
Tvým úkolem je analyzovat zadaný úkol a na základě svých znalostí, zkušeností a rozhodovacích schopností ohodnotit obtížnost úkolu.

#Úkol
{question}

#Formát odpovědi
Obtížnost úkolu ohodnoť na následující pětibodové škále:
- 1: Velmi snadný
- 2: Snadný
- 3: Středně obtížný
- 4: Obtížný
- 5: Velmi obtížný

Vždy vrať jen a pouze příslušné hodnocení ve formě textu (např. "1: Velmi snadný").
"""


with open("evaluation/cs_wildbench/cs_dataset_diff-v1v2.jsonl", "r") as f:
    data = f.readlines()

data = [json.loads(line) for line in data]

prompts = []
for i, d in enumerate(data):
    prompts.append(prompt_template.format(question=d["questions"]))

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


import pandas as pd

df = pd.read_json("evaluation/cs_wildbench/cs_dataset_diff-v1v2.jsonl", lines=True)
df["DeepSeek-R1_response"] = deepseek_responses

df.to_json("evaluation/cs_wildbench/cs_dataset_diff-v1v2_deepseek.jsonl", orient="records", lines=True)