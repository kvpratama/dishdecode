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
    logger.info(f"Resizing image: {state.get('image_path', '<missing>')}")
    stream_writer = get_stream_writer()
    try:
        stream_writer({"custom_key": "Processing the menu..."})
        if "image_path" not in state:
            raise KeyError("'image_path' key not found in state.")
        if "max_size" not in state:
            raise KeyError("'max_size' key not found in state.")

        image = Image.open(state["image_path"])
        original_width, original_height = image.size
        if original_width == 0 or original_height == 0:
            raise ValueError("Image has zero width or height.")

        # Determine scale factor
        if original_width > original_height:
            scale = state["max_size"] / float(original_width)
        else:
            scale = state["max_size"] / float(original_height)

        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_image.save(state["image_path"])

        logger.info(f"Resized image: {state['image_path']}")
        return {"image_path": state["image_path"]}
    except Exception as e:
        logger.error(f"Exception in preprocess_image: {e}", exc_info=True)
        if stream_writer:
            stream_writer({"custom_key": f"Error: {str(e)}"})
        return {"image_path": None, "error": str(e)}


def process_image_with_llm(
    *,
    state: GraphState,
    log_message: str,
    stream_message: str,
    parser: JsonOutputParser,
    prompt: str,
    model_name: str,
    postprocess_fn=None,
):
    logger.info(f"{log_message}: {state.get('image_path', '<missing>')}")
    stream_writer = get_stream_writer()
    try:
        stream_writer({"custom_key": stream_message})
        if "image_path" not in state:
            raise KeyError("'image_path' key not found in state.")
        llm = get_llm(model_name=model_name)

        with open(state["image_path"], "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt + f"\n\n{parser.get_format_instructions()}",
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{encoded_image}",
                },
            ]
        )

        response = llm.invoke([human_message])
        parsed = parser.parse(response.content)
        if postprocess_fn:
            return postprocess_fn(parsed, stream_writer)
        return parsed
    except Exception as e:
        logger.error(f"Exception in process_image_with_llm: {e}", exc_info=True)
        if stream_writer:
            stream_writer({"custom_key": f"Error: {str(e)}"})
        return {"error": str(e)}


def check_menu(
    state: GraphState, config: dict
) -> Command[Literal["extract_menu", "__end__"]]:
    def postprocess(parsed, stream_writer):
        try:
            is_menu = parsed["is_menu"]
            update = {"is_menu": is_menu}

            if is_menu:
                logger.info("Image is a restaurant menu written in Korean")
                goto = "extract_menu"
            else:
                logger.info("Image is not a restaurant menu written in Korean")
                goto = "__end__"

            return Command(update=update, goto=goto)
        except Exception as e:
            logger.error(f"Exception in check_menu postprocess: {e}", exc_info=True)
            if stream_writer:
                stream_writer({"custom_key": f"Error: {str(e)}"})
            return Command(update={"is_menu": None, "error": str(e)}, goto="__end__")

    try:
        return process_image_with_llm(
            state=state,
            log_message="Checking menu",
            stream_message="Checking image...",
            parser=JsonOutputParser(pydantic_object=CheckImage),
            prompt="Classify the image as a restaurant menu written in Korean. Return True if it is a restaurant menu written in Korean, False otherwise.",
            model_name="gemma-3-12b-it",
            postprocess_fn=postprocess,
        )
    except Exception as e:
        logger.error(f"Exception in check_menu: {e}", exc_info=True)
        stream_writer = get_stream_writer()
        if stream_writer:
            stream_writer({"custom_key": f"Error: {str(e)}"})
        return Command(update={"is_menu": None, "error": str(e)}, goto="__end__")


def extract_menu(state: GraphState, config: dict):
    def postprocess(parsed, stream_writer):
        try:
            logger.info(f"Menu: {parsed}")

            if "dishes" not in parsed:
                raise KeyError("'dishes' key not found in parsed result.")
            stream_writer({"custom_key": f"Menu extracted {parsed['dishes']}..."})
            return {"menu_korean": list(parsed["dishes"])}
        except Exception as e:
            logger.error(f"Exception in extract_menu postprocess: {e}", exc_info=True)
            if stream_writer:
                stream_writer({"custom_key": f"Error: {str(e)}"})
            return {"menu_korean": [], "error": str(e)}

    try:
        return process_image_with_llm(
            state=state,
            log_message="Extract menu",
            stream_message="Decoding dish name...",
            parser=JsonOutputParser(pydantic_object=KoreanDishes),
            prompt="Extract korean dish name from this image. Return a list of Korean dish names.",
            model_name="gemma-3-12b-it",
            postprocess_fn=postprocess,
        )
    except Exception as e:
        logger.error(f"Exception in extract_menu: {e}", exc_info=True)
        stream_writer = get_stream_writer()
        if stream_writer:
            stream_writer({"custom_key": f"Error: {str(e)}"})
        return {"menu_korean": [], "error": str(e)}


def recommend_dishes(state: GraphState, config: dict):
    logger.info(f"Recommend dishes:")
    stream_writer = get_stream_writer()
    stream_writer({"custom_key": "Picking top dishes..."})

    try:
        llm = get_llm(model_name="gemini-2.5-flash-lite-preview-06-17")
        structured_llm = llm.with_structured_output(RecommendedDishList)

        # Validate that 'menu_korean' exists and is a list
        if "menu_korean" not in state:
            raise KeyError("'menu_korean' key not found in state.")
        if not isinstance(state["menu_korean"], list):
            raise ValueError("'menu_korean' must be a list.")
        if len(state["menu_korean"]) == 0:
            raise ValueError("'menu_korean' list is empty.")

        dishes_to_choose_from = "\n".join(state["menu_korean"])
        response = structured_llm.invoke(
            f"You are given a list of Korean dish names. Return three recommended dishes for tourists. Here is the list:\n\n{dishes_to_choose_from}"
        )

        logger.info(f"Recommended dishes: {response}")
        return {"recommended_dishes": response.recommended_dishes}
    except KeyError as e:
        logger.error(f"KeyError in recommend_dishes: {e}")
        stream_writer({"custom_key": f"Error: {str(e)}"})
        return {"recommended_dishes": [], "error": str(e)}
    except Exception as e:
        logger.error(f"Exception in recommend_dishes: {e}", exc_info=True)
        stream_writer({"custom_key": f"Error: {str(e)}"})
        return {"recommended_dishes": [], "error": str(e)}


def search_dish_image(state: GraphState, config: dict):
    logger.info(f"Searching dish image:")
    stream_writer = get_stream_writer()

    try:
        stream_writer(
            {"custom_key": "Retrieving dish images to enhance visual context..."}
        )
        if "recommended_dishes" not in state:
            raise KeyError("'recommended_dishes' key not found in state.")

        tool = TavilySearch(
            max_results=1,
            topic="general",
            include_images=True,
        )

        image_urls = {}
        for dish in state["recommended_dishes"]:
            # Search for dish image using Korean name
            try:
                results = tool.invoke(dish.korean_name)
                if results and "images" in results and results["images"]:
                    image_urls[dish.korean_name] = results["images"]
                else:
                    image_urls[dish.korean_name] = None
            except Exception as e:
                logger.error(
                    f"Exception searching image for dish '{dish.korean_name}': {e}",
                    exc_info=True,
                )
                image_urls[dish.korean_name] = None
        logger.info("Finished searching dish images")
        return {"dish_images": image_urls}
    except Exception as e:
        logger.error(f"Exception in search_dish_image: {e}", exc_info=True)
        if stream_writer:
            stream_writer({"custom_key": f"Error: {str(e)}"})
        return {"dish_images": {}, "error": str(e)}
