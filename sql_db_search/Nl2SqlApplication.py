import configparser
from llama_index.core import SQLDatabase, VectorStoreIndex, Settings
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from sqlalchemy import create_engine, inspect
from sql_db_search.Nl2SqlPrompts import custom_prompt, response_prompt


class Config:
    """Handles loading and accessing configuration from an INI file."""
    def __init__(self, config_file='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def get(self, section, key):
        """Retrieve the configuration value for a given section and key."""
        return self.config.get(section, key)


class Database:
    """Handles database connections and setup."""
    def __init__(self, config: Config):
        self.db_host = config.get('DATABASE', 'host')
        self.db_user = config.get('DATABASE', 'user')
        self.db_password = config.get('DATABASE', 'password')
        self.db_port = config.get('DATABASE', 'port')
        self.db_name = config.get('DATABASE', 'db_name')
        self.engine = self.create_engine()

    def create_engine(self):
        """Create and return a SQLAlchemy engine."""
        return create_engine(
            f'mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}')

    def get_tables(self):
        """Get all table names from the database."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()


class Nl2SqlEngine:
    """Handles the LlamaIndex setup and query execution."""
    def __init__(self, config: Config, database: Database):
        self.gemini_key = config.get('API', 'gemini_key')
        Settings.llm = Gemini(self.gemini_key)
        Settings.embed_model = GeminiEmbedding(api_key=self.gemini_key)

        tables = database.get_tables()
        sql_database = SQLDatabase(database.engine)

        # Set up SQL Table Node Mapping and Schema
        table_node_mapping = SQLTableNodeMapping(sql_database)
        table_schema_objs = [SQLTableSchema(table_name=table) for table in tables]

        # Create ObjectIndex
        self.obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex
        )

        self.query_engine = SQLTableRetrieverQueryEngine(
            sql_database=sql_database,
            table_retriever=self.obj_index.as_retriever(similarity_top_k=3),
            text_to_sql_prompt=custom_prompt,
            response_synthesis_prompt=response_prompt
        )

    def query(self, prompt: str):
        """Query the LlamaIndex engine and return the response."""
        try:
            return self.query_engine.query(prompt)
        except ValueError as e:
            return "Something went wrong, please try again."


class Nl2SqlApplication:
    """Main application to run the query engine."""
    def __init__(self, config_file='config.ini'):
        self.config = Config(config_file)
        self.database = Database(self.config)
        self.nl2SqlEngine = Nl2SqlEngine(self.config, self.database)

    def run(self):
        """Start the prompt loop for querying."""
        while (prompt := input("Enter a prompt (q to quit): ")) != "q":
            response = self.nl2SqlEngine.query(prompt)
            print(response)


# Initialize and run the application
if __name__ == "__main__":
    app = Nl2SqlApplication()
    app.run()
