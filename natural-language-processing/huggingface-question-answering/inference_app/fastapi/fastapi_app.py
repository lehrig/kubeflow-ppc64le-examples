#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

from transformers import AutoTokenizer
from numpy import argmax
from time import time
import spacy
import wikipedia

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
        pass
    else:
        raise ValueError(f"Backend '{backend}' not supported.")

    return (0, 0)


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
        "answer": tokenizer.decode(inputs["input_ids"][0,argmax(outputs[0]):argmax(outputs[1])+1]),
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


