import streamlit as st

st.title("Dish Decode")

st.markdown("""
This app helps you to decode Korean dish names.
""")
if 'thread_id_selfrag' not in st.session_state:
    uploaded_files = st.file_uploader("Choose images", type=["png", "jpg", "jpeg", "heic", "heif"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file)