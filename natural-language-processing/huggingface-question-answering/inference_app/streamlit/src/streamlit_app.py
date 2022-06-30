#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import requests
from datetime import datetime
from pandas import DataFrame

import utils

backends_list = ["Triton Inference Server", "TorchServe", "TFServing"]

# Page config and CSS styling
st.set_page_config(
        page_title="Kubeflow on Power",
        page_icon="https://raw.githubusercontent.com/kubeflow/kubeflow/master/components/centraldashboard/public/assets/favicon.ico",
        layout="wide")

st.markdown(utils.css, unsafe_allow_html=True)

# Headers and columns
st.title("Kubeflow on Power")
st.header("Question Answering")

st.text_input("Inference service URL:", key="inference_url")

st.markdown("***")

left_column, right_column = st.columns(2)


def answer():
    # if empty URL or input, do nothing
    if st.session_state.user_input == "" or st.session_state.inference_url == "":
        return

    # store question and answer to be displayed later (under text input)
    st.session_state.question = st.session_state.user_input

    st.session_state.answer = requests.post(
            "http://localhost:5000/predict", 
            json={
                "question": st.session_state.question, 
                "backend": backend,
                "inference_url": st.session_state.inference_url}
    ).json()

    # reset question field
    st.session_state.user_input = ""


# def get_backends_status():
#     st.session_state.backends_status = requests.post(
#             "http://localhost:5000/status",
#             json={"backends": backends_list}).json()




with st.sidebar: ##############################################################################
    backend = st.radio("Backend server to use:", backends_list)

    mma_required = st.checkbox("Enable MMA", value=True, disabled=True)
    logs_required = st.checkbox("Show logs")

    st.markdown("***")

    if "answer" in st.session_state and st.session_state.answer["status"] == "success":
        st.metric("Inference time (in seconds)", round(st.session_state.answer["inference_time"], 3))
    else:
        st.metric("Inference time (in seconds)", None)
    

    st.markdown("***")

    # st.button("Check backends status", on_click=get_backends_status)


    # if "backends_status" in st.session_state:
    #     df = DataFrame(
    #             st.session_state.backends_status.items(), 
    #             columns=["Backend", "Status"])
    #     st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)




with left_column: #############################################################################

    # create user text input for question
    st.text_input("", key="user_input", on_change=answer)
    st.button("Submit", on_click=answer)

    # display results (only when answer exists, so after the first run)
    if "answer" in st.session_state:
        if st.session_state.answer["status"] == "success":
            if st.session_state.answer["answer"] == "":
                st.session_state.answer["answer"] = "I don't know."
            st.markdown(utils.user_component % st.session_state.question, unsafe_allow_html=True)
            st.markdown(utils.bot_component % st.session_state.answer["answer"], unsafe_allow_html=True)
        elif st.session_state.answer["message"]:
            st.error(st.session_state.answer["message"])


st.markdown("***")

if logs_required and "answer" in st.session_state:
    st.json(st.session_state.answer)


with right_column: ############################################################################

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

