import configparser
import os
from typing import Optional

from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.settings import Settings


config = configparser.ConfigParser()
config.read("../config.ini")
GOOGLE_API_KEY = config["API"]["gemini_key"]
PINECONE_API_KEY = config["API"]["pinecone_key"]


Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=GOOGLE_API_KEY
)
Settings.llm = Gemini(api_key=GOOGLE_API_KEY)

class GoogleRestaurant(BaseModel):
    """Data model for a Google Restaurant."""
    restaurant: Optional[str] = None
    food: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    hours: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[float] = None
    review: Optional[str] = None
    description: Optional[str] = None
    nearby_tourist_places: Optional[str] = None


prompt_template_str = """\
    Can you summarize what is in the image\
    and return the answer in JSON format with the following structure:
    {
        "restaurant": null or string,
        "food": null or string,
        "location": null or string,
        "category": null or string,
        "hours": null or string,
        "price": null or string,
        "rating": null or float,
        "review": null or string,
        "description": null or string,
        "nearby_tourist_places": null or string
    }
"""



def pydantic_gemini(
        model_name, output_class, image_documents, prompt_template_str
):
    gemini_llm = GeminiMultiModal(
        api_key=GOOGLE_API_KEY, model_name=model_name
    )

    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_class),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=gemini_llm,
        verbose=True,
    )

    response = llm_program()
    return response


google_image_documents = SimpleDirectoryReader(
    "./google_restaurants"
).load_data()

results = []
for img_doc in google_image_documents:
    pydantic_response = pydantic_gemini(
        "models/gemini-1.5-flash-latest",
        GoogleRestaurant,
        [img_doc],
        prompt_template_str,
    )
    results.append(pydantic_response)


from llama_index.core.schema import TextNode

nodes = []
for res in results:
    text_node = TextNode()
    metadata = {}
    for r in res:
        # Set description as text of TextNode
        if r[0] == "description":
            text_node.text = r[1]
        else:
            # Replace None with an empty string or default value
            metadata[r[0]] = r[1] if r[1] is not None else ""
    text_node.metadata = metadata
    nodes.append(text_node)


# Initialize Pinecone using the new method
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "googleimages"
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name
                    , dimension=768
                    , spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
                    )

index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=index)

# Set up storage context with Pinecone
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index with new storage context
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

index.storage_context.persist(persist_dir="google_images_index")
# Grab 5 search results
retriever = VectorIndexRetriever(index=index, similarity_top_k=1)

query_engine = RetrieverQueryEngine(retriever=retriever)

response = query_engine.query(
    "recommend an Orlando restaurant for me and its nearby tourist places"
)
print(response)



