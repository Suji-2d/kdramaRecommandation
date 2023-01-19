import streamlit as st
import pandas as pd
import numpy as np
import RecommendationModel as rm

st.write("""
# K-drama recommender
Get recommend to **k-drama** similar to you favorites!!
""")

titles = st.selectbox(
    'What is your favorite **k-drama**',
    rm.getAllTitlesAvailable())

st.write('You selected:', titles)

if st.button('Get recommendations'):
    df = pd.DataFrame(
    rm.genre_recomm((titles)),
    columns=('Name','Rating'))

    hide_table_row_index = """
        <style>
        thead tr th:first-child {display:none}
        tbody th {display:none}
        </style>
        """
    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(df)
