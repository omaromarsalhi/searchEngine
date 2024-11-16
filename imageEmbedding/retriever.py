import configparser
import time
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

# import time
# start_time = time.time()
# embedding = Settings.embed_model.get_text_embedding("give me a restaurant in orlando")
# print(f"Embedding time: {time.time() - start_time:.2f} seconds")
#
# query_vector = embedding[0]  # Use the generated embedding
# start_time = time.time()
#
# pc = Pinecone(api_key=PINECONE_API_KEY)
#
# pinecone_index = pc.Index("googleimages")
# # Perform the Pinecone query
# response = pinecone_index.query(vector=embedding, top_k=1, include_metadata=True)

# print(f"Pinecone query time: {time.time() - start_time:.2f} seconds")
# print("Query result:", response)
pc = Pinecone(api_key=PINECONE_API_KEY)

pinecone_index = pc.Index("googleimages")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

retriever = VectorIndexRetriever(index=index, similarity_top_k=1,top_k=1)

query_engine = RetrieverQueryEngine(retriever=retriever)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    try:
        start_time = time.time()
        response = query_engine.query(prompt)
        end_time = time.time()
        print(f"Response: {response}")
        print(f"Query time: {end_time - start_time:.2f} seconds")
    except ValueError as e:
        raise