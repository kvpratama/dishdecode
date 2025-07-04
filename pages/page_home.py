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

st.title("DishDecode")

st.markdown("""
🤔 Menu read like a secret code?

🍽️ Ready to crack it and order like a local?

Snap a photo of any menu—anywhere in the world—and let our powerful app do the rest. DishDecode instantly decodes menu items, recommends top-rated dishes, and even shows you mouthwatering images so you know exactly what to expect.

**How it works:**
- 📸 Snap or upload a photo of any menu.
- ⏳ Sit tight—we’ll decode it in seconds.
- 🍽️ Instantly see English dish names, top picks, and mouthwatering images


No more guessing games. No more menu anxiety. Just smarter, tastier decisions!

:information_source: **Notes:** DishDecode currently works with Korean-language menus only.

---

""")

if "thread_id_dishdecode" not in st.session_state:
    # Turn this into two columns
    cols = st.columns(2)
    enable = st.checkbox("Enable camera")
    with cols[0]:
        camera_image = st.camera_input("Take a picture", disabled=not enable)
    with cols[1]:
        uploaded_file = st.file_uploader(
            "Or choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=False
        )

    file_to_process = camera_image if camera_image else uploaded_file

    if file_to_process:
        
        st.session_state["thread_id_dishdecode"] = str(uuid.uuid4())
        with st.spinner("Processing..."):
            paths = []
            # Use the appropriate method to get bytes and extension
            if uploaded_file:
                file_bytes = uploaded_file.read()
                file_suffix = os.path.splitext(uploaded_file.name)[1]
            else:
                file_bytes = camera_image.getvalue()
                file_suffix = ".jpg"  # camera_input returns jpg
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_suffix
            ) as tmp:
                logger.info(f"Saving temporary file: {tmp.name}")
                tmp.write(file_bytes)
                paths.append(tmp.name)
            st.session_state["paths"] = paths
            st.rerun()

if "paths" in st.session_state and "result" not in st.session_state:
    st.image(st.session_state["paths"][0], width=400)

    if st.button("Process image"):
        with st.spinner("Processing..."):
            input_data = {
                "image_path": st.session_state["paths"][0],
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
                st.session_state["result"] = result
        st.rerun()
    
if "result" in st.session_state:
    st.image(st.session_state["paths"][0], width=400)

    if not st.session_state["result"]["is_menu"]:
        st.write("The image is not a restaurant menu written in Korean")
    else:
        for dish in st.session_state["result"]["recommended_dishes"]:
            with st.expander(f"{dish.korean_name} / {dish.english_name}"):
                st.write(dish.description)
                st.write(dish.why)
                # Display multiple images in one row
                image_urls = st.session_state["result"]["dish_images"][dish.korean_name]
                if image_urls:
                    n_cols = min(len(image_urls), 3)
                    cols = st.columns(n_cols)
                    for idx, image_url in enumerate(image_urls[:n_cols]):
                        with cols[idx]:
                            st.image(image_url)

    for path in st.session_state["paths"]:
        try:
            logger.info(f"Removing temporary file: {path}")
            os.remove(path)
        except Exception as e:
            logger.error(f"Failed to remove temporary file: {path}")
            logger.error(str(e))
