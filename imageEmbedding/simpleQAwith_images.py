import configparser
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index import (
     GeminiMultiModal, load_image_urls
)


class ConfigLoader:
    """Handles loading configuration from an INI file."""
    def __init__(self, config_path: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

    def get_api_key(self, service_name: str):
        return self.config["API"].get(service_name)


class ImageQueryHandler:
    """Handles multi-modal queries using Gemini."""
    def __init__(self, api_key: str, model_name: str):
        self.gemini_model = GeminiMultiModal(api_key=api_key, model_name=model_name)
        self.image_documents = []

    def load_images(self, image_urls: list):
        """Loads images from URLs into documents."""
        self.image_documents = load_image_urls(image_urls)

    def query_with_images(self, prompt: str):
        """Queries the Gemini model with a prompt and image documents."""
        if not self.image_documents:
            raise ValueError("No images loaded. Please load images first.")
        return self.gemini_model.complete(prompt, image_documents=self.image_documents)


class InteractiveImageQueryEngine:
    """Interactive interface for image-based queries."""
    def __init__(self, query_handler: ImageQueryHandler):
        self.query_handler = query_handler

    def run(self):
        """Starts an interactive session for multi-modal queries."""
        while (prompt := input("Enter a prompt (q to quit): ")) != "q":
            try:
                response = self.query_handler.query_with_images(prompt)
                print(response)
            except ValueError as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    # Load configuration
    CONFIG_PATH = "../config.ini"
    config_loader = ConfigLoader(CONFIG_PATH)
    GOOGLE_API_KEY = config_loader.get_api_key("gemini_key")

    # Initialize ImageQueryHandler
    IMAGE_URLS = [
        "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg",
    ]
    MODEL_NAME = "models/gemini-1.5-pro-latest"

    query_handler = ImageQueryHandler(api_key=GOOGLE_API_KEY, model_name=MODEL_NAME)
    query_handler.load_images(IMAGE_URLS)

    # Start interactive querying
    interactive_engine = InteractiveImageQueryEngine(query_handler)
    interactive_engine.run()




