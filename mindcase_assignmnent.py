import os
import streamlit as st
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, ServiceContext, load_index_from_storage
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.storage import StorageContext
from llama_index.core.callbacks import CallbackManager

callback_manager = CallbackManager()
# Initialize LlamaCPP with a pretrained model.
llm = LlamaCPP(
    model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    context_window=4096,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": -1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# Load documents
documents = SimpleDirectoryReader(input_files=["./Mindcase Data.pdf"]).load_data()
documents = Document(text="\n\n".join([doc.text for doc in documents]))

# Build vector index
def get_build_index(documents, llm, embed_model="local:BAAI/bge-small-en-v1.5", sentence_window_size=3, save_dir="./vector_store/index"):
    node_parser = SentenceWindowNodeParser(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        index = VectorStoreIndex.from_documents([documents], service_context=sentence_context)
        index.storage_context.persist(persist_dir=save_dir)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir), service_context=sentence_context)
    return index

vector_index = get_build_index(documents=documents, llm=llm, embed_model="local:BAAI/bge-small-en-v1.5", sentence_window_size=3, save_dir="./vector_store/index")

# Create query engine
def get_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=rerank_top_n, model="BAAI/bge-reranker-base")
    engine = sentence_index.as_query_engine(similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank])
    return engine

query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=6, rerank_top_n=2)
