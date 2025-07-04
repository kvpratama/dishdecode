# from langgraph.constants import Send
import os
import logging
import time
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from dishdecode.state import RecommendedDishList, GraphState, KoreanDishes, CheckImage
from PIL import Image
from dishdecode.llm import get_llm

# from prompts import load_prompt
import base64
import json

# from llm_model import get_gemma27b_llm, get_gemma12b_llm
from langgraph.config import get_stream_writer
from typing import Dict, List, Literal
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)


def preprocess_image(state: GraphState, config: dict):
    logger.info(f"Resizing image: {state['image_path']}")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "Processing the menu..."})

    image = Image.open(state["image_path"])
    original_width, original_height = image.size

    # Determine scale factor
    if original_width > original_height:
        scale = state["max_size"] / float(original_width)
    else:
        scale = state["max_size"] / float(original_height)

    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # # save resized image using original filename
    resized_image.save(state["image_path"])
    logger.info(f"Resized image: {state['image_path']}")
    return {"image_path": state["image_path"]}


# TODO: Create a node to check if the image is Restaurant Menu written in Korean


def check_menu(state: GraphState, config: dict) -> Command[Literal["extract_menu", "__end__"]]:
    logger.info(f"Checking menu: {state['image_path']}")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "Checking image..."})

    parser = JsonOutputParser(pydantic_object=CheckImage)

    # Initialize the Gemini model
    llm = get_llm(model_name="gemma-3-12b-it")
    # Load and encode local image
    with open(state["image_path"], "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    plain_prompt = (
        "Classify the image as a restaurant menu written in Korean. Return True if it is a restaurant menu written in Korean, False otherwise."
    )
    human_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": plain_prompt + f"\n\n{parser.get_format_instructions()}",
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encoded_image}",
            },
        ]
    )

    response = llm.invoke([human_message])
    parsed = parser.parse(response.content)
    is_menu = parsed["is_menu"]

    update = {"is_menu": is_menu}
    if is_menu:
        logger.info("Image is a restaurant menu written in Korean")
        goto = "extract_menu"
    else:
        logger.info("Image is not a restaurant menu written in Korean")
        goto = "__end__"

    return Command(update=update, goto=goto)


def extract_menu(state: GraphState, config: dict):
    logger.info(f"Extract menu: {state['image_path']}")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "Decoding dish name..."})

    parser = JsonOutputParser(pydantic_object=KoreanDishes)

    # Initialize the Gemini model
    llm = get_llm(model_name="gemma-3-12b-it")
    # Load and encode local image
    with open(state["image_path"], "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    plain_prompt = (
        "Extract korean dish name from this image. Return a list of Korean dish names."
    )
    human_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": plain_prompt + f"\n\n{parser.get_format_instructions()}",
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{encoded_image}",
            },
        ]
    )

    response = llm.invoke([human_message])
    parsed = parser.parse(response.content)

    logger.info(f"Menu: {parsed}")
    stream_writer({"custom_key": f"Menu extracted {parsed['dishes']}..."})
    return {"menu_korean": list(parsed["dishes"])}


def recommend_dishes(state: GraphState, config: dict):
    logger.info(f"Recommend dishes:")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "Picking top dishes..."})

    llm = get_llm(model_name="gemini-2.5-flash-lite-preview-06-17")
    structured_llm = llm.with_structured_output(RecommendedDishList)

    dishes_to_choose_from = "\n".join(state["menu_korean"])
    response = structured_llm.invoke(
        f"You are given a list of Korean dish names. Return three recommended dishes for tourists. Here is the list:\n\n{dishes_to_choose_from}"
    )

    logger.info(f"Recommended dishes: {response}")
    return {"recommended_dishes": response.recommended_dishes}


def search_dish_image(state: GraphState, config: dict):
    logger.info(f"Searching dish image:")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "Retrieving dish images to enhance visual context..."})

    tool = TavilySearch(
        max_results=1,
        topic="general",
        include_images=True,
    )

    image_urls = {}
    for dish in state["recommended_dishes"]:
        # Search for dish image using Korean name
        result = tool.invoke(dish.korean_name)
        image_urls[dish.korean_name] = result["images"]
    return {"image_urls": image_urls}
