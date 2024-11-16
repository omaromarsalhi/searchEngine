import configparser
from prompts import context
from PIL import Image
from llama_index.core import Settings, QueryBundle
from llama_index.core.agent import ReActAgent
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.indices.multi_modal import MultiModalVectorIndexRetriever
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.tools import FunctionTool
from pinecone import Pinecone


# Helper function to display an image
def show_image(url):
    img = Image.open(url)
    img.show()


# Load API keys
config = configparser.ConfigParser()
config.read("../config.ini")
GOOGLE_API_KEY = config["API"]["gemini_key"]
PINECONE_API_KEY = config["API"]["pinecone_key"]

DATA_PATH = "fruit_images/"

# Initialize LlamaIndex settings
Settings.llm = Gemini(GOOGLE_API_KEY)
Settings.embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY)
Settings.chunk_size = 512

# Set up Pinecone
pinecone_client = Pinecone(PINECONE_API_KEY)
index_name = "images"

pinecone_index = pinecone_client.Index(index_name)
image_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
)
# storage_context = StorageContext.from_defaults(image_store=image_store)
# Load and index data
# documents = SimpleDirectoryReader(DATA_PATH).load_data()
# index = MultiModalVectorStoreIndex.from_vector_store(vector_store=image_store)
# index = MultiModalVectorStoreIndex.from_vector_store(
#     vector_store=vector_store
# )
# index = MultiModalVectorStoreIndex.from_vector_store(
#     vector_store=vector_store
# )
# Load the MultiModalVectorStoreIndex
index = MultiModalVectorStoreIndex.from_vector_store(vector_store=image_store, image_vector_store=image_store)

retriever = MultiModalVectorIndexRetriever(
    index=index, image_similarity_top_k=1
)


# Define tool 1: Retrieve images based on text input
def text_to_image_tool(prompt: str):
    results = retriever.text_to_image_retrieve(prompt)
    filtered_results = [result for result in results if result.score >= 0.1]
    for result in filtered_results:
        print("score: " + str(result.score))
        show_image(result.metadata.get('file_path'))
    return "Images displayed based on text input."


# Define tool 2: Retrieve images based on image input
def image_to_image_tool(image_path: str):
    query_bundle = QueryBundle(query_str="", image_path=image_path)
    results = retriever.image_to_image_retrieve(query_bundle)
    filtered_results = [result for result in results if result.score >= 0.5]
    for result in filtered_results:
        print("score: " + str(result.score))
        show_image(result.metadata.get('file_path'))
    return "Images displayed based on image input."


# Create tools for the LlamaIndex agent
tools = [
    FunctionTool.from_defaults(name="RetrieveImagesFromText", fn=text_to_image_tool,
                               description="Retrieve images based on text input."),
    FunctionTool.from_defaults(name="RetrieveImagesFromImage", fn=image_to_image_tool,
                               description="Retrieve images based on image input.")
]

# Initialize LlamaIndex agent
agent = ReActAgent.from_tools(tools, verbose=True, context=context)

# Interactive interface for the agent
while (user_input := input("Enter your query: ")) != "q":
    try:
        print(agent.chat(user_input))
    except Exception as e:
        print(f"Error: {e}")
