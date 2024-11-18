import configparser
from PIL import Image
from llama_index.core import Settings, StorageContext, SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


class Config:
    """Handles loading and accessing configuration from an INI file."""
    def __init__(self, config_file='../config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def get(self, section, key):
        """Retrieve the configuration value for a given section and key."""
        return self.config.get(section, key)


class PineCone:
    def __init__(self, config: Config):
        self.documents = None
        self.gemini_key = config.get('API', 'gemini_key')
        self.pinecone_key = config.get('API', 'pinecone_key')
        self.pinecone_index_name = config.get('API', 'pinecone_index_name')

        Settings.llm = Gemini(self.gemini_key)
        Settings.embed_model = GeminiEmbedding(api_key=self.gemini_key)

        pinecone_client = Pinecone(self.pinecone_key)

        if self.pinecone_index_name not in pinecone_client.list_indexes().names():
            pinecone_client.create_index(self.pinecone_index_name
                                         , dimension=512
                                         , spec=ServerlessSpec(cloud="aws",
                                                               region="us-east-1")
                                         )

        pinecone_index = pinecone_client.Index(self.pinecone_index_name)
        image_store = PineconeVectorStore(pinecone_index=pinecone_index)

        self.storage_context = StorageContext.from_defaults(
            image_store=image_store
        )

    def load_documents(self, data_path):
        self.documents = SimpleDirectoryReader(data_path).load_data()
        keys_to_keep = ["file_name", "file_path", "file_type"]
        for doc in self.documents:
            metadata = doc.extra_info or {}
            filtered_metadata = {key: metadata[key] for key in keys_to_keep if key in metadata}
            doc.extra_info = filtered_metadata

    def save_embeddings(self):
        MultiModalVectorStoreIndex.from_documents(
            self.documents,
            storage_context=self.storage_context,
        )

class ImageEmbeddingApplication:
    """Main application to run the query engine."""
    def __init__(self, config_file,images_path):
        self.config = Config(config_file)
        self.pinecone = PineCone(self.config)
        self.images_path = images_path

    def run(self):
        self.pinecone.load_documents("fruit_images/")
        self.pinecone.save_embeddings()


if __name__ == "__main__":
    app = ImageEmbeddingApplication('../config.ini',"fruit_images/")
    app.run()


