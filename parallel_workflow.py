import os 
from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph ,START,END
from langchain_groq import ChatGroq


llm=ChatGroq(
    model="openai/gpt-oss-120b", api_key=""
)
def merge_score_dicts(existing:dict,newupdate:dict)->dict:
    if existing is None:
        return newupdate
    return {**existing, **newupdate}

# Creating a class

class AnalyzerState(TypedDict):
    raw_text:str
    safety_score:Annotated[dict[str,int],merge_score_dicts]


def toxictiy_node(state:AnalyzerState)->dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response=llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under our single state key
    return {"safety_scores": {"toxicity_level": score}}

def copy_write_node(state:AnalyzerState):

    print("\n [Branch 2] Analyzing Copy_write content")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response=llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"copyright_risk": score}}

def culture_node(state: AnalyzerState) -> dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"cultural_insensitivity": score}}


builder=StateGraph(AnalyzerState)

builder.add_node("toxic",toxictiy_node)
builder.add_node("copywrite",copy_write_node)
builder.add_node("culture",culture_node)


builder.add_edge(START,"toxic")
builder.add_edge(START,"copywrite")
builder.add_edge(START,"culture")


builder.add_edge("toxic",END)
builder.add_edge("copywrite",END)
builder.add_edge("culture",END)

app=builder.compile()


sample_script = """
    Yo guys! Welcome back to the stream. Today I am going to show you how to hack into 
    your friend's system using a script I copied directly from an online forum. 
    Honestly, traditional security protocols are absolute garbage and anyone still using 
    them is an absolute idiot. Let's dive into the code!
    """


initial_state={
    "raw_text":sample_script,
    "safety_score":{}
}

final_state=app.invoke(initial_state)

print(final_state['safety_score'])