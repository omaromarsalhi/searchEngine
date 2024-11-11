from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.multi_modal_llms.gemini import GeminiMultiModal

image_urls = [
    "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg",
]

image_documents = load_image_urls(image_urls)

gemini_pro = GeminiMultiModal(model_name="models/gemini-1.5-pro-latest")

complete_response = gemini_pro.complete(
    prompt="Identify the city where this photo was taken.",
    image_documents=image_documents,
)

print(complete_response)


# stream_complete_response = gemini_pro.stream_complete(
#     prompt="Give me more context for this image",
#     image_documents=image_documents,
# )
#
# for r in stream_complete_response:
#     print(r.text, end="")