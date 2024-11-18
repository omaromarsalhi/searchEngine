import configparser
from prompts import context, prompt
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

class ConfigLoader:
    """Handles loading configuration from an INI file."""
    def __init__(self, config_path: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def get_api_key(self, service_name: str):
        return self.config["API"].get(service_name)


class ImageDisplay:
    """Handles image display functionality."""
    @staticmethod
    def show_image(url: str):
        img = Image.open(url)
        img.show()


class LlamaIndexInitializer:
    """Initializes LlamaIndex settings and tools."""
    def __init__(self, google_api_key: str, pinecone_api_key: str, data_path: str, index_name: str):
        self.google_api_key = google_api_key
        self.pinecone_api_key = pinecone_api_key
        self.data_path = data_path
        self.index_name = index_name

        self._initialize_llama_settings()
        self.vector_store = self._setup_pinecone()
        self.retriever = self._initialize_retriever()

    def _initialize_llama_settings(self):
        Settings.llm = Gemini(self.google_api_key)
        Settings.embed_model = GeminiEmbedding(api_key=self.google_api_key)
        Settings.chunk_size = 512

    def _setup_pinecone(self):
        pinecone_client = Pinecone(self.pinecone_api_key)
        pinecone_index = pinecone_client.Index(self.index_name)
        return PineconeVectorStore(pinecone_index=pinecone_index)

    def _initialize_retriever(self):
        index = MultiModalVectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, image_vector_store=self.vector_store
        )
        return MultiModalVectorIndexRetriever(index=index, image_similarity_top_k=3)

    def create_tools(self):
        return [
            FunctionTool.from_defaults(
                name="RetrieveImagesFromText",
                fn=self.text_to_image_tool,
                description="Retrieve images based on text input."
            ),
            FunctionTool.from_defaults(
                name="RetrieveImagesFromImage",
                fn=self.image_to_image_tool,
                description="Retrieve images based on image input."
            )
        ]

    def text_to_image_tool(self, prompt: str):
        results = self.retriever.text_to_image_retrieve(prompt)
        unique_results = list({result.node_id: result for result in results}.values())
        filtered_results = [result for result in unique_results if result.score >= 0.1]
        for result in filtered_results:
            print("score: " + str(result.score))
            ImageDisplay.show_image(result.metadata.get('file_path'))
        return "Images displayed based on text input."

    def image_to_image_tool(self, image_path: str):
        query_bundle = QueryBundle(query_str="", image_path=image_path)
        results = self.retriever.image_to_image_retrieve(query_bundle)
        unique_results = list({result.node_id: result for result in results}.values())
        filtered_results = [result for result in unique_results if result.score >= 0.9]
        for result in filtered_results:
            print("score: " + str(result.score))
            ImageDisplay.show_image(result.metadata.get('file_path'))
        return "Images displayed based on image input."


class InteractiveAgent:
    """Handles user interaction with the LlamaIndex agent."""
    def __init__(self, tools, prompt_context: str):
        self.agent = ReActAgent.from_tools(tools, verbose=True, context=prompt_context)

    def run(self):
        while (user_input := input("Enter your query: ")) != "q":
            try:
                print(self.agent.chat(user_input))
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    # Load configuration
    config_loader = ConfigLoader("../config.ini")
    GOOGLE_API_KEY = config_loader.get_api_key("gemini_key")
    PINECONE_API_KEY = config_loader.get_api_key("pinecone_key")

    # Initialize the application
    DATA_PATH = "fruit_images/"
    INDEX_NAME = "images"
    llama_initializer = LlamaIndexInitializer(GOOGLE_API_KEY, PINECONE_API_KEY, DATA_PATH, INDEX_NAME)

    # Create tools for the agent
    tools = llama_initializer.create_tools()
    prompt_context = "Welcome to the image retrieval system."
    # Start interactive agent
    interactive_agent = InteractiveAgent(tools, prompt_context)
    interactive_agent.run()

