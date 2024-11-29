import configparser
import os
from typing import Any
from google.generativeai.types import content_types, generation_types
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.llms.gemini import Gemini
from llama_index.llms.gemini.utils import completion_from_gemini_response
import google.generativeai as genai

from agentsOrchestration.test_hitl_agent.MyGeminiModel import MyGeminiModel

config = configparser.ConfigParser()
config.read("../../config.ini")
os.environ["GOOGLE_API_KEY"] = config.get('API', 'gemini_key')



genai.configure(api_key=config.get('API', 'gemini_key'))

def get_order_status(order_id: str) -> str:
    """Fetches the status of a given order ID."""
    # Mock data for example purposes
    order_statuses = {
        "12345": "Shipped",
        "67890": "Processing",
        "11223": "Delivered"
    }
    return order_statuses.get(order_id, "Order ID not found.")


def initiate_return(order_id: str, reason: str) -> str:
    """Initiates a return for a given order ID with a specified reason."""
    if order_id in ["12345", "67890", "11223"]:
        return f"Return initiated for order {order_id} due to: {reason}."
    else:
        return "Order ID not found. Cannot initiate return."


model = genai.GenerativeModel(
    model_name='gemini-1.5-flash-latest',
    # tools=[get_order_status, initiate_return] # list of all available tools
)
response = model.generate_content(
    "What is the status of order 12345?",
    tools=[get_order_status, initiate_return]
)
print(response.candidates[0].content.parts[0].function_call)

#
# llm = MyGeminiModel()
# chat_message = ChatMessage(role="user", content="What is the status of order 12345?")
# response =  llm.achat_with_tools(
#     chat_message,
#     tools=[get_order_status, initiate_return],
# )
#
# print(response)
# print(response.candidates[0].content.parts[0].function_call)

# print(response)
# Assuming `response` is the object returned by the Google Generative AI API
