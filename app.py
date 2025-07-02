import streamlit as st

st.set_page_config(page_title="Dish Decode", page_icon=":material/chef_hat:")

page_home = st.Page("./pages/page_home.py", title="Home", icon=":material/home:")

pg = st.navigation([page_home])

pg.run()