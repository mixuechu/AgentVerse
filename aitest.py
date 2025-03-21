import os
from openai import AzureOpenAI

client = AzureOpenAI(
  api_key = "f317dfd5256942ad873d3e13a1eb1dc7",
  api_version = "2024-08-01-preview",
  azure_endpoint = "https://exbq.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview")

response = client.chat.completions.create(
    model="gpt-4o-mini", # model = "deployment_name".
    messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": "Who were the founders of Microsoft?"}
    ]
)

#print(response)
# print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)