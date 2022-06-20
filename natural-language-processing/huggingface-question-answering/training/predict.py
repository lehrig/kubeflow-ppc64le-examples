#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from transformers import AutoTokenizer
from onnxruntime import InferenceSession
from numpy import argmax
import time

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
session = InferenceSession(
        "onnx/checkpoint-2739/model.onnx",
        providers=["CPUExecutionProvider"])

# print("Inputs: ", [model_input.name  for model_input  in session.get_inputs()])
# print("Outputs:", [model_output.name for model_output in session.get_outputs()])

import json
with open("ex1.json", "r") as f:
    ex1 = json.load(f)

example = {
    "context": "Architecturally, the school has a Catholic character. Atop the Main Building\"s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
    "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
    "answers": {
        "text": ["Saint Bernadette Soubirous"],
        "answer_start": [515]
        }
    }


inputs = tokenizer(example["question"], example["context"], return_tensors="np")

start = time.time()
outputs = session.run(
        output_names=["start_logits", "end_logits"], 
        input_feed=dict(inputs))
end = time.time()

answer = tokenizer.decode(inputs["input_ids"][0,argmax(outputs[0]):argmax(outputs[1])+1])

print(f"Answer: {answer}")
print(f"Inference time: {end - start:.3f} s")
