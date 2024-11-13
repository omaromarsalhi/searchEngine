import configparser
from llama_index.core import Settings, StorageContext,SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


config = configparser.ConfigParser()
config.read("../config.ini")
GOOGLE_API_KEY = config["API"]["gemini_key"]
PINECONE_API_KEY = config["API"]["pinecone_key"]

DATA_PATH= "fruit_images/"


Settings.llm = Gemini(GOOGLE_API_KEY)
Settings.embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY)
Settings.chunk_size = 512


pinecone_client = Pinecone(PINECONE_API_KEY)
index_name = "images"
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(index_name
                                 , dimension=768
                                 , spec=ServerlessSpec(cloud="aws",
                                                       region="us-east-1"
                                                       )
    )

pinecone_index = pinecone_client.Index(index_name)

# Create a PineconeVectorStore using the specified pinecone_index
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Create a StorageContext using the created PineconeVectorStore
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

documents = SimpleDirectoryReader(DATA_PATH).load_data()

index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

retriever = VectorIndexRetriever(index=index, similarity_top_k=1)
query_engine = RetrieverQueryEngine(retriever=retriever)

# Run a sample query
response = query_engine.query("Is there a banana in there?")
print("Query response:", response)
