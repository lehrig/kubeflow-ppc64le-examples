#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import requests
import utils

st.set_page_config(
        page_title="Kubeflow on Power",
        page_icon="https://raw.githubusercontent.com/kubeflow/kubeflow/master/components/centraldashboard/public/assets/favicon.ico",
        layout="wide")

st.markdown(utils.css, unsafe_allow_html=True)

st.title("Kubeflow on Power")
st.header("Question Answering")

left_column, right_column = st.columns(2)


def answer():
    # if empty submit, do nothing
    if st.session_state.user_input == "":
        return

    # store question and answer to be displayed later (under text input)
    st.session_state.question = st.session_state.user_input

    st.session_state.answer = requests.post(
            "http://localhost:5000/", 
            json={"question": st.session_state.question, "backend": backend}).json()

    # reset question field
    st.session_state.user_input = ""


with st.sidebar:
    backend = st.selectbox("Backend server to use:",
            ["TorchServe", "TFServing", "Triton Inference Server"])

    logs_required = st.checkbox("Show logs")


with left_column:

    # create user text input for question
    st.text_input("", key="user_input", on_change=answer)
    st.button("Submit", on_click=answer)

    # display results (only when answer exists, so after the first run)
    if "answer" in st.session_state:
        st.markdown(utils.user_component % st.session_state.question, unsafe_allow_html=True)
        st.markdown(utils.bot_component % st.session_state.answer["answer"], unsafe_allow_html=True)

st.markdown("***")
if logs_required and "answer" in st.session_state:
    st.json(st.session_state.answer)

# Examples column
with right_column:
    examples = [
        "Where did Neil Armstrong study?",
        "When was Miles Davis born?",
        "Who founded Coca-Cola?",
        "With who did Steve Jobs create Apple?",
        "Why did Mandela go to prison?",
        "Where are the United Nations headquarters?",
    ]

    def set_question(example_id):
        st.session_state.user_input = examples[example_id]


    for example_id, example in enumerate(examples):
        st.button(example, on_click=set_question, args=(example_id,))


