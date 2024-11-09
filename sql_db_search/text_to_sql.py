import configparser
from llama_index.core import SQLDatabase, Document, VectorStoreIndex, Settings, QueryBundle
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from openai import api_key
from sqlalchemy import create_engine, inspect
from sql_db_search.prompt import custom_prompt, response_prompt
from vars import get_gemini_api_key


config = configparser.ConfigParser()
config.read('config.ini')

db_host = config['DATABASE']['host']
db_user = config['DATABASE']['user']
db_password = config['DATABASE']['password']
db_port = config['DATABASE']['port']
db_name = config['DATABASE']['db_name']
gemini_key = config['API']['gemini_key']


Settings.llm = Gemini(gemini_key)
Settings.embed_model = GeminiEmbedding(api_key=gemini_key)


engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
inspector = inspect(engine)



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


query_engine = SQLTableRetrieverQueryEngine(
    sql_database=sql_database,
    table_retriever=obj_index.as_retriever(similarity_top_k=3),
    text_to_sql_prompt=custom_prompt,
    response_synthesis_prompt=response_prompt
)


while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    try:
        response = query_engine.query(prompt)
        print(response)
    except ValueError as e:
        print("something went wrong, please try again")

