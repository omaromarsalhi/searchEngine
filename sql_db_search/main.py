# from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
# from llama_index.embeddings.gemini import GeminiEmbedding
# from sqlalchemy import (
#     create_engine,
#     inspect
# )
# from llama_index.core import GPTVectorStoreIndex, Document, SQLDatabase, PromptTemplate
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
# sql_database = SQLDatabase(engine)
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
# # schema_description = ""
# #
# # # Create a detailed schema description
# # for table_name in inspector.get_table_names():
# #     columns = inspector.get_columns(table_name)
# #     column_names = ", ".join([column["name"] for column in columns])
# #     schema_description += f"Table: {table_name} (Columns: {column_names})\n"
# #     schema_document_text = f"Table: {table_name} (Columns: {column_names})"
# # schema_documents.append(Document(text=schema_document_text))
#
# # Create the index with documents containing the schema of your database
# # index = GPTVectorStoreIndex.from_documents(documents)
# # Create a query engine for the index
# # query_engine = index.as_query_engine()
# custom_prompt = PromptTemplate(
#     template=f"{system_prefix}\n\n{examples}\n\nUser input: {{query}}\n\nSQL query:"
# )
#
# # Define a prompt template that limits responses to SQL only
#
#
# query_engine = NLSQLTableQueryEngine(
#     sql_database,
#     tables=tables,
#     text_to_sql_prompt=custom_prompt,
#     index=GPTVectorStoreIndex.from_documents(documents)
# )
#
#
# # Define the function to process user queries
# def process_query(query):
#     response = query_engine.query(query)
#     return response
#
#
# while (query := input("Enter a prompt (q to quit): ")) != "q":
#     sql_query = process_query(query)
#     print(f"Generated SQL: {sql_query}")