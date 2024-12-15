# import configparser
# import time
#
# import pandas as pd
# import functools
# import json
#
# from llama_index.core.base.llms.types import ChatMessage, MessageRole
# from llama_index.core.tools import FunctionTool
# from llama_index.llms.mistralai import MistralAI
#
# config = configparser.ConfigParser()
# config.read("../config.ini")
#
#
# # Step 1: Data Preparation
# data = {
#     'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
#     'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
#     'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
#     'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
#     'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
# }
#
# # Step 2: Define Tools
# def retrieve_payment_status(transaction_id: str) -> str:
#     df = pd.DataFrame(data)
#     if transaction_id in df.transaction_id.values:
#         return json.dumps({'status': df[df.transaction_id == transaction_id].payment_status.item()})
#     return json.dumps({'error': 'Transaction ID not found.'})
#
#
# def retrieve_payment_date(transaction_id: str) -> str:
#     df = pd.DataFrame(data)
#     if transaction_id in df.transaction_id.values:
#         return json.dumps({'date': df[df.transaction_id == transaction_id].payment_date.item()})
#     return json.dumps({'error': 'Transaction ID not found.'})
#
#
# tools = [
#     FunctionTool.from_defaults(fn=retrieve_payment_status),
#     FunctionTool.from_defaults(fn=retrieve_payment_date),
# ]
#
# model = "mistral-large-latest"
# client = MistralAI(api_key=config.get('API', 'mistral_key'))
#
# # Step 3: Correct Interaction with the Model
# # Initial query message
# messages = [ChatMessage(role=MessageRole.USER, content="What's the status of my transaction T1001?")]
#
# # First interaction
# response = client.chat_with_tools(
#     chat_history=messages,
#     tools=tools,
# )
#
# # Check if tool_calls exist in the response
# if "tool_calls" not in response.message.additional_kwargs or not response.message.additional_kwargs["tool_calls"]:
#     raise ValueError("Tool calls missing in the first response. Ensure the model is configured properly.")
#
# # Extract the tool_call_id
# tool_call_id = response.message.additional_kwargs["tool_calls"][0].id
# if not tool_call_id:
#     raise ValueError("Tool call ID is missing in the response.")
#
# # Add the AI response and the corresponding tool response to the message history
# messages += [
#     response.message,
#     ChatMessage(
#         role=MessageRole.TOOL,
#         content="Paid",
#         additional_kwargs={"tool_name": "retrieve_payment_status", "tool_call_id": str(tool_call_id)},
#     ),
# ]
#
# # Second interaction
# response = client.chat_with_tools(
#     chat_history=messages,
#     tools=tools,
# )
#
# # Output the final response
# print("Response content:", response.message.content)
# print("Response additional kwargs:", response.message.additional_kwargs)
# print("Response role:", response.message.role)
#
# # print(response.message.content)
# print(response.message.additional_kwargs)
# print(response.message.role)


import configparser
import time

import pandas as pd
import functools
import json

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool
from llama_index.llms.mistralai import MistralAI
import uuid

from MyTestingOrchestrator.MyMistralAI import MyMistralAI

config = configparser.ConfigParser()
config.read("../config.ini")


class MyChatMessage(ChatMessage):
    tool_call_id: str | None = None
    name: str | None = None


# Step 1: Data Preparation
data = {
    'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
    'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
    'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
}


# Step 2: Define Tools
def retrieve_payment_status(transaction_id: str) -> str:
    df = pd.DataFrame(data)
    if transaction_id in df.transaction_id.values:
        return json.dumps({'status': df[df.transaction_id == transaction_id].payment_status.item()})
    return json.dumps({'error': 'Transaction ID not found.'})


def retrieve_payment_date(transaction_id: str) -> str:
    df = pd.DataFrame(data)
    if transaction_id in df.transaction_id.values:
        return json.dumps({'date': df[df.transaction_id == transaction_id].payment_date.item()})
    return json.dumps({'error': 'Transaction ID not found.'})


tools = [
    FunctionTool.from_defaults(fn=retrieve_payment_status),
    FunctionTool.from_defaults(fn=retrieve_payment_date),
]

model = "mistral-large-latest"
client = MistralAI(api_key=config.get('API', 'mistral_key'))

# Step 3: Correct Interaction with the Model
# Initial query message
messages = [ChatMessage(role=MessageRole.USER, content="What's the status of my transaction T1001?")]

# First interaction
response = client.chat_with_tools(
    chat_history=messages,
    tools=tools,
)

# Generate a dummy tool call ID if no tool calls are suggested.
if "tool_calls" not in response.message.additional_kwargs or not response.message.additional_kwargs["tool_calls"]:
    tool_call_id = str(uuid.uuid4())  # generating a random ID
    response.message.additional_kwargs["tool_calls"] = [
        {"id": tool_call_id, "function": {}, "tool_name": ""}]  # dummy tool call object

else:
    # print(response.message.additional_kwargs["tool_calls"][0].function.name)
    tool_call_id = response.message.additional_kwargs["tool_calls"][0].id
    function_name = response.message.additional_kwargs["tool_calls"][0].function.name
time.sleep(5)

# Add the AI response and the corresponding tool response to the message history. If a tool isn't used, this will be an empty call with our dummy tool_call_id.
# messages += [
#     response.message,
#     MyChatMessage(
#         role=MessageRole.TOOL,
#         content='{"status": "Paid"}',  # This is dummy content, replace with logic if necessary. It's not used by the model.
#         tool_call_id=str(tool_call_id),
#         name="retrieve_payment_status"
#     ),
# ]

messages += [
    response.message,
    ChatMessage(
        role=MessageRole.TOOL,
        content='{"status": "Paid"}',
        additional_kwargs={"tool_name": "retrieve_payment_status", "tool_call_id": str(tool_call_id)},
    ),
]
print(str(messages))

# Second interaction
# response = client.chat_with_tools(
#     user_msg=MyChatMessage(
#         role=MessageRole.USER,
#         content='{"status": "Paid"}',
#         tool_call_id=str(tool_call_id),
#         name="retrieve_payment_status"
#     ),
#     chat_history=messages,
#     tools=tools,
# )
response = client.chat_with_tools(
    # user_msg=ChatMessage(
    #     role=MessageRole.TOOL,
    #     content='{"status": "Paid"}',
    #     additional_kwargs={"tool_name": "retrieve_payment_status", "tool_call_id": str(tool_call_id)},
    # ),
    chat_history=messages,
    tools=tools,
)

# Output the final response
print("Response content:", response)
# print("Response content:", response.message.content)
# print("Response additional kwargs:", response.message.additional_kwargs)
# print("Response role:", response.message.role)
