from typing import List
from langgraph.graph import MessagesState
# import operator

# Define the state type with annotations
class GraphState(MessagesState):
    image_path: str
    max_size: int
    menu_korean: List[str]
    menu_english: List[str]
    menu_description: List[str]
    menu_images: List[str]

class GraphStateInput(MessagesState):
    image_path: str
    max_size: int

class GraphStateOutput(MessagesState):
    menu_korean: List[str]
    menu_english: List[str]
    menu_description: List[str]
    menu_images: List[str]