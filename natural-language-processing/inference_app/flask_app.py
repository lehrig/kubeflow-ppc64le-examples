#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request

from transformers import AutoTokenizer
from onnxruntime import InferenceSession
from numpy import argmax
from time import time
import spacy
import wikipedia

app = Flask(__name__)


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

session = InferenceSession(
        "../onnx/checkpoint-2739/model.onnx",
        providers=["CPUExecutionProvider"])

nlp = spacy.load("en_core_web_sm")


@app.route('/predict', methods=["POST"])
def run_inference():
    question = request.json["question"]
    context = wikipedia.summary(nlp(question).ents[0].text)
    inputs = tokenizer(question, context,
            return_tensors="np", truncation=True)

    outputs = session.run(
            output_names=["start_logits", "end_logits"],
            input_feed=dict(inputs))

    return {"answer": tokenizer.decode(inputs["input_ids"][0,argmax(outputs[0]):argmax(outputs[1])+1])}


if __name__ == '__main__':
    app.run()


