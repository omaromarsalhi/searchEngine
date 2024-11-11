import os
from typing import Optional

from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
import matplotlib.pyplot as plt
from PIL import Image
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.settings import  Settings



GOOGLE_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
PINECONE_API_KEY = ""

Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=GOOGLE_API_KEY
)
Settings.llm = Gemini(api_key=GOOGLE_API_KEY)

# image_urls = [
#     "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg",
# ]
#
# image_documents = load_image_urls(image_urls)
#
# gemini_pro = GeminiMultiModal(model_name="models/gemini-1.5-pro-latest")
#
# img_response = requests.get(image_urls[0])
# print(image_urls[0])
# img = Image.open(BytesIO(img_response.content))


# complete_response = gemini_pro.complete(
#     prompt="Identify the city where this photo was taken.",
#     image_documents=image_documents,
# )
#
# print(complete_response)


# stream_complete_response = gemini_pro.stream_complete(
#     prompt="Give me more context for this image",
#     image_documents=image_documents,
# )
#
# for r in stream_complete_response:
#     print(r.text, end="")

# from pathlib import Path
#
# input_image_path = Path("google_restaurants")
# if not input_image_path.exists():
#     Path.mkdir(input_image_path)
#
#
# import requests
# import os
#
# # Directory to save images
# os.makedirs('./google_restaurants', exist_ok=True)
#
# urls = {
#     "miami": "https://docs.google.com/uc?export=download&id=1Pg04p6ss0FlBgz00noHAOAJ1EYXiosKg",
#     "orlando": "https://docs.google.com/uc?export=download&id=1dYZy17bD6pSsEyACXx9fRMNx93ok-kTJ",
#     "sf": "https://docs.google.com/uc?export=download&id=1ShPnYVc1iL_TA1t7ErCFEAHT74-qvMrn",
#     "toronto": "https://docs.google.com/uc?export=download&id=1WjISWnatHjwL4z5VD_9o09ORWhRJuYqm"
# }
#
# for city, url in urls.items():
#     response = requests.get(url)
#     if response.status_code == 200:
#         with open(f'./google_restaurants/{city}.png', 'wb') as f:
#             f.write(response.content)
#         print(f"{city.capitalize()} image downloaded successfully.")
#     else:
#         print(f"Failed to download {city} image.")

#
# class GoogleRestaurant(BaseModel):
#     """Data model for a Google Restaurant."""
#     restaurant: Optional[str] = None
#     food: Optional[str] = None
#     location: Optional[str] = None
#     category: Optional[str] = None
#     hours: Optional[str] = None
#     price: Optional[str] = None
#     rating: Optional[float] = None
#     review: Optional[str] = None
#     description: Optional[str] = None
#     nearby_tourist_places: Optional[str] = None
#
#
# prompt_template_str = """\
#     Can you summarize what is in the image\
#     and return the answer in JSON format with the following structure:
#     {
#         "restaurant": null or string,
#         "food": null or string,
#         "location": null or string,
#         "category": null or string,
#         "hours": null or string,
#         "price": null or string,
#         "rating": null or float,
#         "review": null or string,
#         "description": null or string,
#         "nearby_tourist_places": null or string
#     }
# """
#
#
#
# def pydantic_gemini(
#         model_name, output_class, image_documents, prompt_template_str
# ):
#     gemini_llm = GeminiMultiModal(
#         api_key=GOOGLE_API_KEY, model_name=model_name
#     )
#
#     llm_program = MultiModalLLMCompletionProgram.from_defaults(
#         output_parser=PydanticOutputParser(output_class),
#         image_documents=image_documents,
#         prompt_template_str=prompt_template_str,
#         multi_modal_llm=gemini_llm,
#         verbose=True,
#     )
#
#     response = llm_program()
#     return response
#
#
# google_image_documents = SimpleDirectoryReader(
#     "./google_restaurants"
# ).load_data()
#
# results = []
# for img_doc in google_image_documents:
#     pydantic_response = pydantic_gemini(
#         "models/gemini-1.5-flash-latest",
#         GoogleRestaurant,
#         [img_doc],
#         prompt_template_str,
#     )
#     results.append(pydantic_response)
#
#
# from llama_index.core.schema import TextNode
#
# nodes = []
# for res in results:
#     text_node = TextNode()
#     metadata = {}
#     for r in res:
#         # Set description as text of TextNode
#         if r[0] == "description":
#             text_node.text = r[1]
#         else:
#             # Replace None with an empty string or default value
#             metadata[r[0]] = r[1] if r[1] is not None else ""
#     text_node.metadata = metadata
#     nodes.append(text_node)


# Initialize Pinecone using the new method
# pc = Pinecone(api_key=PINECONE_API_KEY)
#
# index_name = "googleimages"
# if index_name not in pc.list_indexes().names():
#     pc.create_index(index_name
#                     , dimension=768
#                     , spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#                     )

# index = pc.Index(index_name)
# vector_store = PineconeVectorStore(pinecone_index=index)
#
# # Set up storage context with Pinecone
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
#
# # Create index with new storage context
# index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
#
# # Grab 5 search results
# retriever = VectorIndexRetriever(index=index, similarity_top_k=1)
#
# query_engine = RetrieverQueryEngine(retriever=retriever)
#
# # Now you query:
#
# llm_query = query_engine.query("recommend an Orlando restaurant for me and its nearby tourist places")
# print(llm_query.response)

# Inspect results
# print([i.get_content() for i in answer])
# query_engine = index.as_query_engine(
#     similarity_top_k=1,
# )
#
# response = query_engine.query(
#     "recommend an Orlando restaurant for me and its nearby tourist places"
# )
# print(response)





# Create a Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Get the Pinecone index
pinecone_index = pc.Index("googleimages")

# Initialize the PineconeVectorStore with the Pinecone index
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Create a StorageContext with the PineconeVectorStore
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load your documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

# Create the VectorStoreIndex with the documents and the StorageContext
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)