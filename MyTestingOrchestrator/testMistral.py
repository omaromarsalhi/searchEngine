import configparser
import pandas as pd
import functools
import json
import os

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool
from llama_index.llms.mistralai import MistralAI

config = configparser.ConfigParser()
config.read("../config.ini")

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

# Corrected 'ChatMessage' usage
messages = ChatMessage(role=MessageRole.USER, content="What's the status of my transaction T1001?")

response = client.chat_with_tools(
    user_msg=messages,
    tools=tools,
)
print(response)
