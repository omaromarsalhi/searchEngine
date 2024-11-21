import os
import pandas as pd
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.experimental import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.gemini import Gemini

from noteEngine import note_engine
from pdf import pdf_reader
from prompts import context, instruction_str, new_prompt
from vars import *

population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})


# Set up LLM
Settings.llm = Gemini(api_key=get_gemini_api_key(), temperature=0)

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
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_query_engine",
            description="Provides detailed information about the world population.",
        ),
    ),
]

# Initialize ReActAgent without predefined context since we will update it dynamically
agent = ReActAgent.from_tools(tools, verbose=True, context=context)

# Query loop with memory-enhanced context
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    try:
        result = agent.chat(prompt)
        print(result)
    except ValueError as e:
        if str(e) == "Reached max iterations.":
            print("Maximum iterations reached. Restarting the process...")
        else:
            raise
