# from langgraph.constants import Send
import os
import logging
import time
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from dishdecode.state import GraphStateInput
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
# from prompts import load_prompt
import base64
import json
# from llm_model import get_gemma27b_llm, get_gemma12b_llm
from langgraph.config import get_stream_writer
from typing import Dict, List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def preprocess_image(state: GraphStateInput, config: dict):
    logger.info(f"Resizing image: {state['image_path']}")
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
    resized_image.save(os.path.splitext(state["image_path"])[0] + "_resized.jpg")
    logger.info(f"Resized image: {os.path.splitext(state["image_path"])[0] + "_resized.jpg"}")
    return {"image_path": os.path.splitext(state["image_path"])[0] + "_resized.jpg"}


class KoreanDish(BaseModel):
    """Simplified model using dictionary for all translations"""
    dishes: List[str] = Field(
        description="Korean dish names extracted from the image"
    )


def extract_menu(state: GraphStateInput, config: dict):
    logger.info(f"Extract menu: {state['image_path']}")

    parser = JsonOutputParser(pydantic_object=KoreanDish)

    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemma-3-12b-it",
        temperature=0,
        max_tokens=None,
        timeout=60,  # Added a timeout
        max_retries=2,
    )
    # Load and encode local image
    with open(state["image_path"], "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    plain_prompt = "Extract korean dish name from this image. Return a list of Korean dish names."
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": plain_prompt + f"\n\n{parser.get_format_instructions()}"},
            # {"type": "text", "text": plain_prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
        ]
    )

    response = llm.invoke([human_message])
    parsed = parser.parse(response.content)
    logger.info(f"Menu: {parsed}")
    return {"menu_korean": list(parsed["dishes"])}