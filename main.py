from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
from noteEngine import note_engine
from pdf import pdf_reader
from prompts import context
from vars import *


Settings.llm = Gemini(api_key=get_gemini_api_key(),temperature=0.7)

# Define tools
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=pdf_reader,
        metadata=ToolMetadata(
            name="transistor_reader",
            description="Provides detailed information about computer transistors.",
        ),
    ),
]

# Create the ReActAgent with your custom LLM and tools
agent = ReActAgent.from_tools(tools, verbose=True, context=context)

# Query loop
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    try:
        result = agent.query(prompt)
        print(result)
    except ValueError as e:
        if str(e) == "Reached max iterations.":
            print("Maximum iterations reached. Restarting the process...")
        else:
            raise




# from llama_index.core.tools import QueryEngineTool, ToolMetadata
# from llama_index.core.agent import ReActAgent
# from llama_index.core.settings import Settings
# from llama_index.llms.ollama import Ollama
# from pdf import pdf_reader
# from prompts import  context
# from vars import *
#
#
#
#
# Settings.llm=Ollama(base_url=get_ollama_url(),
#                     model=get_ollama_model())
#
# # Define tools
# tools = [
#     QueryEngineTool(
#         query_engine=pdf_reader,
#         metadata=ToolMetadata(
#             name="transistor_reader",
#             description="this gives detailed information about the transistors the computer component",
#         ),
#     ),
# ]
#
# # Create the ReActAgent with your custom LLM
# agent = ReActAgent.from_tools(tools, verbose=True, context=context)
#
#
# # Query loop
# while (prompt := input("Enter a prompt (q to quit): ")) != "q":
#     try:
#         result = agent.query(prompt)
#         print(result)
#     except ValueError as e:
#         if str(e) == "Reached max iterations.":
#             print("Maximum iterations reached. Restarting the process...")
#         else:
#             raise
#     # result = agent.query(prompt)
#     # print(result)
