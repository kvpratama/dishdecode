from typing import List, Dict
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
# import operator


class CheckImage(BaseModel):
    """Check if the image is Restaurant Menu written in Korean"""

    is_menu: bool = Field(description="Is the image a restaurant menu?")


class KoreanDishes(BaseModel):
    """Korean dish names extracted from the image"""

    dishes: List[str] = Field(description="Korean dish names extracted from the image")


class RecommendedDish(BaseModel):
    """Simplified model using dictionary for all translations"""

    korean_name: str = Field(description="Dish name in Korean")
    english_name: str = Field(
        description="Literal English translation of the dish name"
    )
    description: str = Field(description="Description of the dish in English")
    why: str = Field(description="Why this dish is recommended for tourists")


class RecommendedDishList(BaseModel):
    recommended_dishes: List[RecommendedDish] = Field(
        description="List of recommended dishes"
    )


# Define the state type with annotations
class GraphState(MessagesState):
    image_path: str
    max_size: int
    menu_korean: List[str]
    recommended_dishes: List[RecommendedDish]
    dish_images: Dict[str, list[str]]
    is_menu: bool


class GraphStateInput(MessagesState):
    image_path: str
    max_size: int


class GraphStateOutput(MessagesState):
    menu_korean: List[str]
    recommended_dishes: List[RecommendedDish]
    dish_images: Dict[str, list[str]]
    is_menu: bool
