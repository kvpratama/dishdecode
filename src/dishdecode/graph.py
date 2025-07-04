from langgraph.graph import START, END, StateGraph
from dishdecode.state import GraphState, GraphStateInput, GraphStateOutput
from dishdecode.nodes import (
    preprocess_image,
    check_menu,
    extract_menu,
    recommend_dishes,
    search_dish_image,
)
from langgraph.checkpoint.memory import MemorySaver
# from dishdecode.configuration import ConfigSchema

builder = StateGraph(GraphState, input=GraphStateInput, output=GraphStateOutput)
builder.add_node("preprocess_image", preprocess_image)
builder.add_node("check_menu", check_menu)
builder.add_node("extract_menu", extract_menu)
builder.add_node("recommend_dishes", recommend_dishes)
builder.add_node("search_dish_image", search_dish_image)

builder.add_edge(START, "preprocess_image")
builder.add_edge("preprocess_image", "check_menu")
builder.add_edge("extract_menu", "recommend_dishes")
builder.add_edge("recommend_dishes", "search_dish_image")
builder.add_edge("search_dish_image", END)

graph = builder.compile()
