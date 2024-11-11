# Install dependencie
# !pip install llama-index 'google-generativeai>=0.3.0' matplotlib pinecone-client
import configparser
import os
from pathlib import Path
import random
from typing import Optional, List

from PIL import Image as PILImage
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.indices.vector_store import VectorIndexRetriever, VectorIndexAutoRetriever
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreInfo, MetadataInfo
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field
from advanced_ocr.prompts import  custom_receipts_prompt


config = configparser.ConfigParser()
config.read("config.ini")
GOOGLE_API_KEY = config["API"]["gemini_key"]
PINECONE_API_KEY = config["API"]["pinecone_key"]

Settings.embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY)
Settings.llm = Gemini(api_key=GOOGLE_API_KEY)


def get_image_files(dir_path, sample: Optional[int] = 10, shuffle: bool = False):
    dir_path = Path(dir_path)
    image_paths = list(dir_path.glob("*.jpg"))
    if shuffle:
        random.shuffle(image_paths)
    return image_paths[:sample] if sample else image_paths


# image_files = get_image_files("SROIE2019/test/img", sample=5)
image_files = get_image_files("myTest", sample=2)
print(image_files)


class ReceiptInfo(BaseModel):
    company: Optional[str] = Field(None, description="Company name")
    date: Optional[str] = Field(None, description="Date in DD/MM/YYYY")
    address: Optional[str] = Field(None, description="Address")
    total: Optional[float] = Field(None, description="Total amount")
    currency: Optional[str] = Field(None, description="Currency abbreviation")
    summary: Optional[str] = Field(None, description="Summary of the receipt")


prompt_template_str = """\
    Please summarize the details of the receipt in a short paragraph, including:
    - Company name
    - Date
    - Address
    - Total amount
    - Currency
    Also, return this information in a structured JSON format as shown below:

    {
        "company": "Company Name",
        "date": "DD/MM/YYYY",
        "address": "Address",
        "total": 0.0,
        "currency": "Currency",
        "summary": "Summary of the receipt"
    }
"""


def pydantic_gemini(output_class, image_documents, prompt_template_str):
    gemini_llm = GeminiMultiModal(api_key=GOOGLE_API_KEY, model_name="models/gemini-1.5-flash-latest")
    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_class),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        multi_modal_llm=gemini_llm,
        verbose=True,
    )
    return llm_program()


def process_image_file(image_file):
    print(f"Processing {image_file}")
    img_docs = SimpleDirectoryReader(input_files=[image_file]).load_data()
    return pydantic_gemini(ReceiptInfo, img_docs, prompt_template_str)


outputs = [process_image_file(img) for img in image_files]


def get_nodes_from_objs(objs: List[ReceiptInfo], image_files: List[str]) -> List[TextNode]:
    nodes = []
    for image_file, obj in zip(image_files, objs):
        summary = obj.summary if obj.summary is not None else ""
        company = obj.company if obj.company is not None else ""
        date = obj.date if obj.date is not None else ""
        address = obj.address if obj.address is not None else ""
        total = obj.total if obj.total is not None else 0.0
        currency = obj.currency if obj.currency is not None else ""

        node = TextNode(
            text=summary,
            metadata={
                "company": company,
                "date": date,
                "address": address,
                "total": total,
                "currency": currency,
                "image_file": str(image_file)
            },
            excluded_embed_metadata_keys=["image_file"],
            excluded_llm_metadata_keys=["image_file"],
        )
        nodes.append(node)
    return nodes


nodes = get_nodes_from_objs(outputs, image_files)

# Initialize Pinecone using the new method
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "omar3"
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

vector_store_info = VectorStoreInfo(
    content_info="Receipts",
    metadata_info=[
        MetadataInfo(name="company", description="Store name", type="string"),
        MetadataInfo(name="address", description="Store address", type="string"),
        MetadataInfo(name="date", description="Purchase date in DD/MM/YYYY", type="string"),
        MetadataInfo(name="total", description="Total amount", type="float"),
        MetadataInfo(name="currency", description="Country currency abbreviation", type="string"),
    ],
)


retriever = VectorIndexAutoRetriever(
    index=index,
    vector_store_info=vector_store_info,
    similarity_top_k=2,
    empty_query_top_k=10,
    verbose=True,
    prompt_template_str=custom_receipts_prompt.template
)



def display_response(nodes: List[TextNode]):
    """Display response with images."""
    for node in nodes:
        print(node.get_content(metadata_mode="all"))
        # image_path = node.metadata.get("image_file")
        # if image_path and os.path.exists(image_path):
        #     print(f"Displaying image: {image_path}")
        #     img = PILImage.open(image_path)
        #     img.show()
        # else:
        #     print(f"Image file not found: {image_path}")


while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    try:
        retriever.update_prompts({"custom_receipts_prompt": custom_receipts_prompt})
        nodes = retriever.retrieve(prompt)
        display_response(nodes)

    except ValueError as e:
        if str(e) == "Reached max iterations.":
            print("Maximum iterations reached. Restarting the process...")
        else:
            raise
