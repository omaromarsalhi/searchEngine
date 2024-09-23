import os
import shutil
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from vars import *


Settings.embed_model = OllamaEmbedding(base_url=get_ollama_url(),
                                       model_name=get_ollama_model())

Settings.llm = Ollama(base_url=get_ollama_url(),
                      model=get_ollama_model())


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        # shutil.rmtree(index_name)
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


pdf_path = os.path.join("data", "Canada.pdf")
canada_pdf = PDFReader().load_data(file=pdf_path)
canada_index = get_index(canada_pdf, "canada")
canada_engine = canada_index.as_query_engine()
