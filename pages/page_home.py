import streamlit as st
import os
import tempfile
import logging
import uuid
from dishdecode.graph import graph

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

st.title("Dish Decode")

st.markdown("""
This app helps you to decode Korean dish names.
""")
if "thread_id_dishdecode" not in st.session_state:
    uploaded_file = st.file_uploader(
        "Choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )

    if uploaded_file:
        st.image(uploaded_file, width=400)
        st.session_state["thread_id_dishdecode"] = str(uuid.uuid4())
        with st.spinner("Processing..."):
            paths = []
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp:
                logger.info(f"Saving temporary file: {tmp.name}")
                tmp.write(uploaded_file.read())
                paths.append(tmp.name)

            input_data = {
                "image_path": paths[0],
                "max_size": 640,
            }
            with st.empty():
                for stream_mode, chunk in graph.stream(
                    input_data,
                    config={
                        "configurable": {
                            "thread_id": st.session_state["thread_id_dishdecode"]
                        }
                    },
                    stream_mode=["values", "custom"],
                ):
                    if stream_mode == "custom":
                        st.write(chunk.get("custom_key", ""))
                    elif stream_mode == "values":
                        result = chunk
                        st.write("")

            for dish in result["recommended_dishes"]:
                with st.expander(f"{dish.korean_name} / {dish.english_name}"):
                    st.write(dish.description)
                    st.write(dish.why)
                    # Display multiple images in one row
                    image_urls = result["image_urls"][dish.korean_name]
                    if image_urls:
                        n_cols = min(len(image_urls), 3)
                        cols = st.columns(n_cols)
                        for idx, image_url in enumerate(image_urls[:n_cols]):
                            with cols[idx]:
                                st.image(image_url)

            for path in paths:
                try:
                    logger.info(f"Removing temporary file: {path}")
                    os.remove(path)
                except Exception as e:
                    logger.error(f"Failed to remove temporary file: {path}")
                    logger.error(str(e))
