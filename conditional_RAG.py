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

    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)

    chunks=splitter.split(documents)

    vectorstores=FAISS.from_documents(chunks,embedding)

    return vectorstores.as_retriever(search_kwargs={"k":4})


academic_retriver=build_retriever("academics_handbook.pdf")
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
    retrieved_context:str


def classifier_node(state:pipeline)->dict:
    """Look at the latest user message and decide which path to take."""

    last_message=state['message'][-1].content


    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response=llm.invoke(prompt)
    category=response.content.strip().lower()

    if "academic" in category:
        category="academic"

    elif "fee" in category:
        category="fee"

    else:
        category="general"

    return {"query_type":category}

def academic_rag_node(state:pipeline)->dict:
    """Retrieves relevant chunks from the academics handbook."""
    query=state["message"][-1].content
    docs = academic_retriver.invoke(query)
    context="\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context":context}

def fee_rag_node(state:pipeline)->dict:
    """Retrieves relevant chunks from the academics handbook."""
    query=state["message"][-1].content
    docs = fee_retriever.invoke(query)
    context="\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context":context}

def general_node(state:pipeline)->dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

def response_node(state:pipeline)->dict:
    """Generates the final answer , personalised using the students programme""" 
    query=state["message"][-1].content
    programme=state.get("programme","unknown")
    context=state["retrieved_context"]

    if context=="NO_RETRIEVAL_NEEDED":
        prompt=(
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )

    else:
        prompt=(
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."           
        )

    response=llm.invoke(prompt)
    return {"message":[("ai",response.content.strip())]}


# STEP 4 --> CREATING Router function

def route_state(state:pipeline):
    if state["query_type"]=="academic":
        return "academic_rag"
    elif state["query_type"]=="fee":
        return "fee_rag"
    else:
        return "general"
    
# Step 5 - Building the graph

graph=StateGraph(pipeline)

graph.add_node("classifier",classifier_node)
graph.add_node("academic_rag",academic_rag_node)
graph.add_node("fee_rag",fee_rag_node)
graph.add_node("general",general_node)
graph.add_node("response",response_node)


# Edges

graph.add_edge(START,"classifier")

graph.add_conditional_edges(
    "classifier",route_state
)


graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("general","response")

graph.add_edge("response",END)

app = graph.compile()

# STEPS 6 -RUN THE CODE

print("WELCOME TO THE COLLEGE ASSISTANT")

print("WHICH PROGRAMME ARE YOU IN")
print("1. BCA")
print("2. BBA")
print("3. B.com")


choice=input("\n Enter 1,2 ,3 ")

programe_map={
    "1":"BCA",
    "2":"BBA",
    "3":"B.COM"
}

student_proramme=programe_map.get(choice,"BBA")

    
print(f"\n Great ! You're set as a {student_proramme} student")


while True:
    user_query=input("You: ")

    if user_query.lower() in ["exit","quit"]:
        break

    result=app.invoke({
        "programme":student_proramme,
        "message":[("human",user_query)]
    })

    print(f"Assistant:{result['message'][-1].content}")






























































