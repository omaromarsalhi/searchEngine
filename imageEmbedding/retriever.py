import configparser
import time
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core import  VectorStoreIndex
from llama_index.core.settings import Settings




class ConfigLoader:
    """Handles loading configuration from an INI file."""
    def __init__(self, config_path: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def get_api_key(self, service_name: str):
        return self.config["API"].get(service_name)


class LlamaIndexInitializer:
    """Initializes LlamaIndex settings and components."""
    def __init__(self, google_api_key: str, pinecone_api_key: str, index_name: str):
        self.google_api_key = google_api_key
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name

        self._initialize_llama_settings()
        self.vector_store = self._setup_pinecone()
        self.query_engine = self._initialize_query_engine()

    def _initialize_llama_settings(self):
        """Sets up LlamaIndex with embedding and language model settings."""
        Settings.embed_model = GeminiEmbedding(
            model_name="models/embedding-001", api_key=self.google_api_key
        )
        Settings.llm = Gemini(api_key=self.google_api_key)

    def _setup_pinecone(self):
        """Initializes Pinecone and sets up the vector store."""
        pinecone_client = Pinecone(api_key=self.pinecone_api_key)
        pinecone_index = pinecone_client.Index(self.index_name)
        return PineconeVectorStore(pinecone_index=pinecone_index)

    def _initialize_query_engine(self):
        """Initializes the retriever and query engine."""
        index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=1)
        return RetrieverQueryEngine(retriever=retriever)

    def get_query_engine(self):
        return self.query_engine


class InteractiveQueryEngine:
    """Handles user interaction with the query engine."""
    def __init__(self, query_engine):
        self.query_engine = query_engine

    def run(self):
        """Starts the interactive session for querying."""
        while (prompt := input("Enter a prompt (q to quit): ")) != "q":
            try:
                start_time = time.time()
                response = self.query_engine.query(prompt)
                end_time = time.time()
                print(f"Response: {response}")
                print(f"Query time: {end_time - start_time:.2f} seconds")
            except ValueError as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    # Load configuration
    config_loader = ConfigLoader("../config.ini")
    GOOGLE_API_KEY = config_loader.get_api_key("gemini_key")
    PINECONE_API_KEY = config_loader.get_api_key("pinecone_key")

    # Initialize LlamaIndex and query engine
    INDEX_NAME = "googleimages"
    llama_initializer = LlamaIndexInitializer(GOOGLE_API_KEY, PINECONE_API_KEY, INDEX_NAME)
    query_engine = llama_initializer.get_query_engine()

    # Start interactive querying
    interactive_engine = InteractiveQueryEngine(query_engine)
    interactive_engine.run()

