import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader
from llama_index.core.settings import Settings
from llama_index.embeddings.ollama import OllamaEmbedding 
from llama_index.llms.ollama import Ollama

load_dotenv()

Settings.embed_model=OllamaEmbedding(base_url=os.getenv('OLLAMA_BASE_URL'),
                                     model_name=os.getenv('OLLAMA_MODEL'))


Settings.llm=Ollama(base_url=os.getenv('OLLAMA_BASE_URL'),
                    model=os.getenv('OLLAMA_MODEL'))

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


pdf_path = os.path.join("data", "omar_cv.pdf")
omar_resume_pdf = PDFReader().load_data(file=pdf_path)
omar_resume_index = get_index(omar_resume_pdf, "omar_resume")
omar_resume_engine = omar_resume_index.as_query_engine()
