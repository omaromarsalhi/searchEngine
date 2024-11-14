import configparser

from llama_index.core import PromptTemplate
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.multi_modal_llms.gemini import GeminiMultiModal

config = configparser.ConfigParser()
config.read("../config.ini")
GOOGLE_API_KEY = config["API"]["gemini_key"]


image_urls = [
    "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg",
]


qa_tmpl_str = (
    "Given the images provided, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

qa_tmpl = PromptTemplate(qa_tmpl_str)

image_documents = load_image_urls(image_urls)

gemini_pro = GeminiMultiModal(api_key=GOOGLE_API_KEY, model_name="models/gemini-1.5-pro-latest")
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    try:
        response = gemini_pro.complete(
            prompt,
            image_documents=image_documents,
        )
        print(response)
    except ValueError as e:
        raise

# stream_complete_response = gemini_pro.stream_complete(
#     prompt="Give me more context for this image",
#     image_documents=image_documents,
# )
#
# for r in stream_complete_response:
#     print(r.text, end="")
