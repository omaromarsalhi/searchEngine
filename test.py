
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(model="my model", api_base="https://8514-35-234-52-129.ngrok-free.app/api/generate", api_key="fake")

response = llm.complete("Hello World!")
print(str(response))



# import requests
# import json

# # Replace with your actual URL and endpoint
# base_url = 'https://276d-104-197-159-60.ngrok-free.app'

# def get_completion(prompt):
#     response = requests.post(
#         f"{base_url}/api/generate",  # Adjust the endpoint as needed
#         json={
#             "model": "llama3",
#             "messages": [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": "Who won the world series in 2020?"},
#                 {"role": "assistant", "content": "The LA Dodgers won in 2020."},
#                 {"role": "user", "content": "Where was it played?"}
#             ],
#             "prompt": prompt
#         },stream=True
#     )
#     response.raise_for_status()  # Raise an error for bad responses
#     return response.content.decode('utf-8')

# # Fetch and print the response
# response_json = get_completion(prompt="What is the summary of this document?")
# def finetune_resp(text):
#     newResp=text.split('\n')
#     del newResp[-1]
#     final_response=''
#     for i in newResp:
#         body = json.loads(i)
#         final_response+=body.get('response', '')
#     return final_response
# print(finetune_resp(response_json))
# # print(response_json['choices'][0]['message']['content'])
