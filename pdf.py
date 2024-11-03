import os
import shutil
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.readers.file import PDFReader
from llama_index.core.settings import Settings
from vars import *

# configure(api_key=get_gemini_api_key())
GOOGLE_API_KEY = get_gemini_api_key()

Settings.embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY)

Settings.llm = Gemini(GOOGLE_API_KEY)


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


# Load PDF data
pdf_path = os.path.join("data", "Transistor.pdf")
pdf_data = PDFReader().load_data(file=pdf_path)

# Build or load the index
pdf_index = get_index(pdf_data, "Transistor")
pdf_reader = pdf_index.as_query_engine()

# import os
# import shutil
# from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
# from llama_index.readers.file import PDFReader
# from llama_index.core.settings import Settings
# from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.llms.ollama import Ollama
#
# from vars import *
#
#
# Settings.embed_model = OllamaEmbedding(base_url=get_ollama_url(),
#                                        model_name=get_ollama_model())
#
# Settings.llm = Ollama(base_url=get_ollama_url(),
#                       model=get_ollama_model())
#
#
# def get_index(data, index_name):
#     index = None
#     if not os.path.exists(index_name):
#         # shutil.rmtree(index_name)
#         index = VectorStoreIndex.from_documents(data, show_progress=True)
#         index.storage_context.persist(persist_dir=index_name)
#     else:
#         index = load_index_from_storage(
#             StorageContext.from_defaults(persist_dir=index_name)
#         )
#
#     return index
#
#
# pdf_path = os.path.join("data", "Transistor.pdf")
# canada_pdf = PDFReader().load_data(file=pdf_path)
# canada_index = get_index(canada_pdf, "Transistor")
# pdf_reader = canada_index.as_query_engine()
