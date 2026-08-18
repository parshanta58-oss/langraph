# State is used to maintain and store the data and wrokflow 
# creating a graph
# Create state


# First way to create state : Using Typed dictionaries

from typing import TypedDict

class State(TypedDict):
    topic:str
    summary:str
    score:int


# 2nd way is using pydantic maodel approach 
# Good at data validation and type checking in runtime

from pydantic import BaseModel, field_validator

class State(BaseModel):
    topic:str
    score:int
    summary:str=""

    @field_validator
    def score_positve(cls,v):
        if(v<0):
            raise ValueError("SCORE MUST BE POSITIVE")

# Python Data classes 
# Standard python data class :used rarely

from dataclasses import dataclass,field

@dataclass
class State():
    topic:str
    summary:str=""
    message:list=field(default_factory=list)


# Fourth way is to create by langgraph 

from langgraph.graph import MessageState

class State(MessageState):
    user_name:str
    language:str 

