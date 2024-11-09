from llama_index.core import SQLDatabase, Document, VectorStoreIndex, Settings, QueryBundle
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from sqlalchemy import create_engine, inspect
from sql_db_search.prompt import custom_prompt, response_prompt
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

# Initialize the SQL database and query engine
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

