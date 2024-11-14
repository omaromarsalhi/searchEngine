import configparser
import os

from PIL import Image
from llama_index.core import Settings, StorageContext, SimpleDirectoryReader, QueryBundle
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.indices.multi_modal import MultiModalVectorIndexRetriever
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


def show_image(url):
    img = Image.open(url)
    img.show()


config = configparser.ConfigParser()
config.read("../config.ini")
GOOGLE_API_KEY = config["API"]["gemini_key"]
PINECONE_API_KEY = config["API"]["pinecone_key"]

DATA_PATH = "fruit_images/"

Settings.llm = Gemini(GOOGLE_API_KEY)
Settings.embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY)
Settings.chunk_size = 512

pinecone_client = Pinecone(PINECONE_API_KEY)
index_name = "images"
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(index_name
                                 , dimension=512
                                 , spec=ServerlessSpec(cloud="aws",
                                                       region="us-east-1"
                                                       )
                                 )

pinecone_index = pinecone_client.Index(index_name)
#
# # Create a PineconeVectorStore using the specified pinecone_index
image_store = PineconeVectorStore(pinecone_index=pinecone_index)
#
# # Create a StorageContext using the created PineconeVectorStore
storage_context = StorageContext.from_defaults(
    image_store=image_store
)
#
documents = SimpleDirectoryReader(DATA_PATH).load_data()

index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

retriever = MultiModalVectorIndexRetriever(
    index=index,
    similarity_top_k=1,
    image_similarity_top_k=1
)
# Initialize the retriever
# retriever = index.as_retriever(similarity_top_k=1)

# Interactive loop for querying
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    try:
        # query_bundle = QueryBundle(query_str="", image_path=prompt)
        # response = retriever.image_to_image_retrieve(query_bundle)
        results = retriever.text_to_image_retrieve(prompt)
        filtered_results = [result for result in results if result.score >= 0.5]
        print(filtered_results)
        for resp in filtered_results:
            show_image(resp.metadata.get('file_path'))
    except ValueError as e:
        print("Error:", e)

# retriever_engine = index.as_retriever(image_similarity_top_k=1)
#
# retrieval_results = retriever_engine.image_to_image_retrieve(
#     DATA_PATH + "image_1.jpg"
# )
# print(retrieval_results)

# retriever = VectorIndexRetriever(index=index, similarity_top_k=1)
# query_engine = RetrieverQueryEngine(retriever=retriever)
#
# retriever = VectorIndexRetriever(index=index)
#
# # Run a sample query
#
# while (prompt := input("Enter a prompt (q to quit): ")) != "q":
#     try:
#         response = retriever.retrieve(prompt)
#         print("Query response:", response)
#     except ValueError as e:
#         raise
