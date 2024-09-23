
import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
from note_engine import note_engine
from pdf import canada_engine
from resume import omar_resume_engine
from prompts import new_prompt, instruction_str, context


load_dotenv()

Settings.llm=Ollama(base_url=os.getenv('OLLAMA_BASE_URL'),
                    model=os.getenv('OLLAMA_MODEL'))


# Path to your population data
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)


population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

# Define tools
tools = [
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information at the world population and demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="this gives detailed information about canada the country",
        ),
    ),
    QueryEngineTool(
        query_engine=omar_resume_engine,
        metadata=ToolMetadata(
            name="omar_resume_data",
            description="this gives detailed information about the software engineer omar salhi as it his resume",
        ),
    ),
    # note_engine
]

# Create the ReActAgent with your custom LLM
agent = ReActAgent.from_tools(tools, verbose=True, context=context)


# Query loop
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)

# what percent of canadian speak english as thier first language