import configparser

import pandas as pd
import functools
import json
import os
from mistralai import Mistral

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

df = pd.DataFrame(data)

# Step 2: Define Tools
def retrieve_payment_status(df, transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values:
        return json.dumps({'status': df[df.transaction_id == transaction_id].payment_status.item()})
    return json.dumps({'error': 'Transaction ID not found.'})

def retrieve_payment_date(df, transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values:
        return json.dumps({'date': df[df.transaction_id == transaction_id].payment_date.item()})
    return json.dumps({'error': 'Transaction ID not found.'})

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_status",
            "description": "Get payment status of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_date",
            "description": "Get payment date of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    }
]

names_to_functions = {
    'retrieve_payment_status': functools.partial(retrieve_payment_status, df=df),
    'retrieve_payment_date': functools.partial(retrieve_payment_date, df=df)
}


model = "mistral-large-latest"

client = Mistral(api_key=config.get('API', 'mistral_key'))

# Step 4: User Query and Model Interaction
messages = [{"role": "user", "content": "What's the status of my transaction T1001?"}]

response = client.chat.complete(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="any",
)
print(response)

# Extract tool call information
tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_params = json.loads(tool_call.function.arguments)

# Execute the chosen function
function_result = names_to_functions[function_name](**function_params)

# Step 5: Generate Final Response
# messages.append({"role": "user", "name": function_name, "content": function_result, "tool_call_id": tool_call.id})
messages.append({"role": "user",  "content": function_result, })

final_response = client.chat.complete(
    model=model,
    messages=messages
)

# Output final message
print(final_response.choices[0].message.content)
