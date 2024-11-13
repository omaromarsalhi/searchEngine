import configparser
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings


config = configparser.ConfigParser()
config.read("../config.ini")
GOOGLE_API_KEY = config["API"]["gemini_key"]
PINECONE_API_KEY = config["API"]["pinecone_key"]


Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=GOOGLE_API_KEY
)
Settings.llm = Gemini(api_key=GOOGLE_API_KEY)


pc = Pinecone(api_key=PINECONE_API_KEY)

pinecone_index = pc.Index("googleimages")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader("google_images_index").load_data()

index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

retriever = VectorIndexRetriever(index=index, similarity_top_k=1)

query_engine = RetrieverQueryEngine(retriever=retriever)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    try:
        response = query_engine.query(prompt)
        print(response)
    except ValueError as e:
        raise