from typing import TypedDict

class pipelinestate(TypedDict):
    raw_input:str
    edited_text:str
    script_text:str
    final_output:str


from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

llm=ChatGroq(model="openai/gpt-oss-120b", api_key="")

def editor_node(state:pipelinestate)->dict:
    """Stage 1: cleans up grammer ,remove typos and refines the tone."""

    prompt=(
        "you are an expert copyeditor. Clean up the following raw text."
        "Fix any grammatical errors ,spelling mistakes , and smooth out transition"
        "while keeping the core message intact . Return only the edited text "
        f"Text:\n{state['raw_input']}"
        
    )

    response=llm.invoke(prompt)

    return {"edited_text" : response.content.strip()}


def script_node(state:pipelinestate)->dict:
    """Stage 2 :Format the clean text into an engaging video script style."""
    
    print("\n---[stage 2] Executing ScriptWriter Node ---")

    prompt=(
        "you are  a charismatic Yotube content creator . Take this edited text and transform it into engaging video script"
        "it into a highly engaging .punchy, conversation videp script hook. Make it sound"
        f"Edited text:\n{state['edited_text']}"
    )

    response=llm.invoke(prompt)
    return {"script_text":response.content.strip()}

def translator_node(state:pipelinestate)->dict:
    """Stage 3: Translate the script into natural flowing Hinglish."""
    print("\n--- [Stage 3] Executing Hinglish TRANSLATOR NODE ---")

    prompt=(
        "You are an expert content localizer for the Indian market. Take the following script "
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
        "or repeat information. Alternating comfortably between Hindi and English phrases just like "
        "an intellectual tech educator would speak naturally on a live stream. Keep the energy high! "
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['script_text']}"
    )

    response=llm.invoke(prompt)
    return {"final_output":response.content.strip()}


# State and Nodes are ready , Create a graph , for creating a graph i have to connect nodes , for connecting nodes edges are used

from langgraph.graph import StateGraph, START , END

# Created the graph 

graph=StateGraph(pipelinestate)

# Add the nodes in our graph

graph.add_node("editor",editor_node)
graph.add_node("script_writer",script_node)
graph.add_node("translator",translator_node)


# ADD EDGES (SEQUENTIAL -ONE AFTER ANOTHER)


graph.add_edge(START,"editor")
graph.add_edge("editor","script_writer")
graph.add_edge("script_writer","translator")
graph.add_edge("translator",END)


# Compile the graph 

app=graph.compile()

result=app.invoke(
    {"raw_input":"AI agents are the future of tech. They can think, plan, and act on their own. LangGraph helps you build these agents with proper control and memory."}
)


print("Your result are: - \n\n")
print(result['final_output'])