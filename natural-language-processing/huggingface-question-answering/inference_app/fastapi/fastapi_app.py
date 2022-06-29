#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, Request
from typing import List
from pydantic import BaseModel
import uvicorn

import os
import json
import requests
from transformers import AutoTokenizer
from numpy import argmax
from time import time
import spacy
import wikipedia

def get_env(field, default=None):
    return os.environ[field] if field in os.environ else default

ENDPOINTS = {
    "Triton Inference Server": {
        "model":  get_env("TRITON_MODEL_ENDPOINT",  "http://localhost:8000/v2/models/question-answering/infer")
        "status": get_env("TRITON_STATUS_ENDPOINT", "localhost:8000/v2/health/ready")
    }, {
    "TorchServe": {
        "model":  None,
        "status": None
    "TFServing": {
        "model":  None,
        "status": None

    
app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
nlp = spacy.load("en_core_web_sm")


def get_wikipedia_context(entity):
    try:
        return wikipedia.summary(wikipedia.search(entity, results=1), auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.summary(e.options[0], auto_suggest=False)


def run_inference(inputs, backend):
    if backend == "TorchServe":
        pass
    elif backend == "TFServing":
        pass
    elif backend == "Triton Inference Server":
        payload = {
            "inputs": [
                {
                    "name": "attention_mask",
                    "shape": inputs["attention_mask"].shape,
                    "datatype": "INT64",
                    "data": inputs["attention_mask"].tolist()
                }, {
                    "name": "input_ids",
                    "shape": inputs["input_ids"].shape,
                    "datatype": "INT64",
                    "data": inputs["input_ids"].tolist()
                }
            ]
        }

        response = requests.post(ENDPOINTS[backend]["model"], data=json.dumps(payload)).json()
        return {
            "status": "success",
            "answer": {ent["name"]: argmax(ent["data"]) for ent in response["outputs"]}
        }

    raise NotImplementedError(f"Backend '{backend}' not supported.")



class Data(BaseModel):
    question: str
    backend: str

@app.post("/predict")
async def predict(data: Data):
    try:
        doc = nlp(data.question)
        if len(doc.ents) == 0:
            raise ValueError("Spacy did not find any entity in the question '{data.question}'.")

        context = get_wikipedia_context(doc.ents[0].text)
        inputs = dict(tokenizer(data.question, context,
                return_tensors="np", truncation=True))

        start_time = time()
        outputs = run_inference(inputs, data.backend)["answer"]
        inference_time = time() - start_time

        return {
            "status": "success",
            "message": "",
            "answer": tokenizer.decode(inputs["input_ids"][0, outputs["start_logits"]:outputs["end_logits"]+1]),
            "backend": data.backend,
            "inference_time": inference_time,
            "shape_input_ids": inputs["input_ids"].shape,
            "shape_attention_mask": inputs["attention_mask"].shape,
            "context": context,
            "question": data.question,
            "entities": [(ent.text, ent.label_) for ent in doc.ents]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": repr(e)
        }


class BackendList(BaseModel):
    backends: List[str]

@app.post("/status")
def check_backends_status(backends_list: BackendList):
    results = {}
    for b in backends_list.backends:
        try:
            assert(requests.get(ENDPOINTS[b]["status"]).status_code == 200)
            results[b] = "OK"
        except:
            results[b] = "KO"
    return results

    # return {b: "KO" for b in backends_list.backends}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

