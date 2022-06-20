#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

from transformers import AutoTokenizer
from onnxruntime import InferenceSession
from numpy import argmax
from time import time
import spacy
import wikipedia

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
session = InferenceSession(
        "../onnx/checkpoint-2739/model.onnx",
        providers=["CPUExecutionProvider"])
nlp = spacy.load("en_core_web_sm")


class Data(BaseModel):
    question: str
    backend: str

@app.post("/")
async def run_inference(data: Data):
    doc = nlp(data.question)
    context = wikipedia.summary(doc.ents[0].text)
    inputs = dict(tokenizer(data.question, context,
            return_tensors="np", truncation=True))

    start_time = time()
    outputs = session.run(
            output_names=["start_logits", "end_logits"],
            input_feed=inputs)
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


