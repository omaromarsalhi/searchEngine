import configparser

from llama_index.core import download_loader, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

config = configparser.ConfigParser()
config.read("../config.ini")
GOOGLE_API_KEY = config["API"]["gemini_key"]
PINECONE_API_KEY = config["API"]["pinecone_key"]

DATA_URL = "https://www.gettingstarted.ai/how-to-use-gemini-pro-api-llamaindex-pinecone-index-to-build-rag-app"

llm = Gemini(GOOGLE_API_KEY)
embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embed_model
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

BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

loader = BeautifulSoupWebReader()
documents = loader.load_data(urls=[DATA_URL])
print(documents)

# Create a PineconeVectorStore using the specified pinecone_index
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Create a StorageContext using the created PineconeVectorStore
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# Use the chunks of documents and the storage_context to create the index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

query_engine = index.as_query_engine()

gemini_response = query_engine.query("What does the author think about LlamaIndex?")

print(gemini_response)
