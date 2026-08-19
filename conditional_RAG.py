from dotenv import load_dotenv
load_dotenv()
import os
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph ,START ,END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# LOADERS --> TEXT SPLITTER --> EMBEDDING --> VECTOR STORES --> RETIEVER -->LLM

embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_retriever(pdf_path:str):
    loader=PyPDFLoader(pdf_path)
    documents=loader.load()

    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overalp=100)

    chunks=splitter.split(documents)

    vectorstores=FAISS.from_documents(chunks,embedding)

    return vectorstores.as_retriever(search_kwargs={"k":4})


academic_retirever=build_retriever("academics_handbook.pdf")
fee_retriever=build_retriever("fee_structure.pdf")


llm=ChatGroq(
    model="openai/gpt-oss-120b", api_key=""
)

# state


# Loader --> text_splitter --> embeddings-->vector stores --> retriever-->llm


class pipeline(TypedDict):
    programme:str
    message:Annotated[list,add_messages]
    query_type:str
    retriever_context:str


def programme_node(state:pipeline):
    

