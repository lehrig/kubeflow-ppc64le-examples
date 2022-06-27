#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, Request
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


try:
    TRITON_ENDPOINT = os.environ["TRITON_ENDPOINT"]
except KeyError as e:
    raise KeyError(f"Environment variable {e} is required.")


app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
nlp = spacy.load("en_core_web_sm")


def get_wikipedia_context(entity):
    return wikipedia.summary(wikipedia.search(entity, results=1), auto_suggest=False)


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

        response = requests.post(TRITON_ENDPOINT, data=json.dumps(payload)).json()
        return {ent["name"]: argmax(ent["data"]) for ent in response["outputs"]}

    raise ValueError(f"Backend '{backend}' not supported.")


class Data(BaseModel):
    question: str
    backend: str

@app.post("/")
async def predict(data: Data):
    doc = nlp(data.question)
    context = get_wikipedia_context(doc.ents[0].text)
    inputs = dict(tokenizer(data.question, context,
            return_tensors="np", truncation=True))

    start_time = time()
    outputs = run_inference(inputs, data.backend)
    inference_time = time() - start_time

    return {
        "answer": tokenizer.decode(inputs["input_ids"][0, outputs["start_logits"]:outputs["end_logits"]+1]),
        "backend": data.backend,
        "inference_time": inference_time,
        "shape_input_ids": inputs["input_ids"].shape,
        "shape_attention_mask": inputs["attention_mask"].shape,
        "context": context,
        "question": data.question,
        "entities": [(ent.text, ent.label_) for ent in doc.ents]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

