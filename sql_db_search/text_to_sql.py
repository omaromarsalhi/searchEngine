# Ensure all necessary imports are available

from llama_index.core import SQLDatabase, Document, VectorStoreIndex, Settings, QueryBundle
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from llama_index.core.prompts import PromptType
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_TMPL
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from sqlalchemy import create_engine, inspect
from llama_index.core import PromptTemplate

from sql_db_search.prompt import custom_prompt
from vars import get_gemini_api_key

Settings.llm = Gemini(get_gemini_api_key())
Settings.embed_model = GeminiEmbedding(api_key=get_gemini_api_key())
# Set up the engine and inspector
engine = create_engine('mysql+pymysql://root:@localhost:3306/rag_db')
inspector = inspect(engine)

# Extract table and column information
tables = inspector.get_table_names()

sql_database = SQLDatabase(engine)

table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = []
for table in tables:
    table_schema_objs.append((SQLTableSchema(table_name=table)))

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex
)

# additional_instructions = (
#     "\n\nBy default, please limit your query results to 5 entries unless a specific number is mentioned in the question."
#     " If a number is specified, use that number instead of 5."
# )
#
# # Combine the default prompt with your additional instructions
# modified_prompt = DEFAULT_TEXT_TO_SQL_TMPL + additional_instructions
#
# # Create the modified prompt
# custom_prompt = PromptTemplate(
#     modified_prompt,
#     prompt_type=PromptType.TEXT_TO_SQL,
# )

# Initialize the SQL database and query engine
query_engine = SQLTableRetrieverQueryEngine(
    sql_database=sql_database,
    table_retriever=obj_index.as_retriever(similarity_top_k=3),
    text_to_sql_prompt=custom_prompt
)

# # Run a sample query
try:
    response = query_engine.query("List customers")
    print(response)
except AttributeError as e:
    print("Error:", e)

# from llama_index.core.indices.struct_store import NLSQLTableQueryEngine, SQLTableRetrieverQueryEngine
# from llama_index.embeddings.gemini import GeminiEmbedding
# from sqlalchemy import (
#     create_engine,
#     inspect
# )
# from llama_index.core import  Document, SQLDatabase, PromptTemplate, VectorStoreIndex
# from llama_index.llms.gemini import Gemini
# from llama_index.core import Settings
# from sql_db_search.prompt import system_prefix, examples
# from vars import get_gemini_api_key
#
# Settings.llm = Gemini(get_gemini_api_key())
# Settings.embed_model = GeminiEmbedding(api_key=get_gemini_api_key())
#
# # Create an engine to connect to your database
# engine = create_engine('mysql+pymysql://root:@localhost:3306/rag_db')
# # Create an inspector
# inspector = inspect(engine)
# # Extract table information
# tables = inspector.get_table_names()
# print(tables)
# sql_database = SQLDatabase(engine)
#
# # Extract column information for each table
# schema = {}
# for table in tables:
#     columns = inspector.get_columns(table)
#     schema[table] = [column['name'] for column in columns]
#
# # Convert the schema into documents for LlamaIndex
# documents = []
# for table, columns in schema.items():
#     document_text = f"Table: {table} ({', '.join(columns)})"
#     documents.append(Document(text=document_text))
#
#
# # Create the index with documents containing the schema of your database
# index = VectorStoreIndex.from_documents(documents)
#
# query_engine = SQLTableRetrieverQueryEngine(
#     sql_database,table_retriever=index.as_retriever(similarity_top_k=3)
# )
# response = query_engine.query("How many customers we have?")
# print(response)
#
