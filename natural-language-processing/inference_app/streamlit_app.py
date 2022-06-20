#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import requests

def answer():
    # store question and answer to be displayed later (under text input)
    st.session_state.question = st.session_state.user_input
    st.session_state.answer = requests.post(
            "http://localhost:5000/predict", 
            json={"question": st.session_state.question}).json()["answer"]

    # reinitialise question field
    st.session_state.user_input = ""

st.title("Kubeflow on Power")
st.header("Question Answering")

# create user text input for question
st.text_input("", key="user_input", on_change=answer)

# display results (only when answer exists, so after the first run)
if "answer" in st.session_state:
    st.write(st.session_state.question)
    st.write(st.session_state.answer)

