from langgraph.graph import START, END, StateGraph
from dishdecode.state import GraphState, GraphStateInput, GraphStateOutput
from dishdecode.nodes import preprocess_image, extract_menu, recommend_dishes
from langgraph.checkpoint.memory import MemorySaver
# from dishdecode.configuration import ConfigSchema

builder = StateGraph(GraphState, input=GraphStateInput, output=GraphStateOutput)
builder.add_node("preprocess_image", preprocess_image)
builder.add_node("extract_menu", extract_menu)
builder.add_node("recommend_dishes", recommend_dishes)

builder.add_edge(START, "preprocess_image")
builder.add_edge("preprocess_image", "extract_menu")
builder.add_edge("extract_menu", "recommend_dishes")
builder.add_edge("recommend_dishes", END)

graph = builder.compile()