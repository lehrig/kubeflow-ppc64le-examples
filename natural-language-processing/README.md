# Kubeflow on Power
## NLP Demo

Question Answering example, based on HuggingFace Transformers and trained of SQuAD dataset

Sources: 

* https://huggingface.co/docs/transformers/main/en/tasks/question_answering
* https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb


### Environment setup

```
conda config --prepend channels https://ftp.osuosl.org/pub/open-ce/1.6.1/
conda config --prepend channels huggingface

conda create -n opence \
	pytorch=1.10.2=cuda11.2_py39_1 \
	datasets=2.3.1 \
	transformers=4.18.0 \
	onnxruntime=1.11.0
```

### Start the inference app

* StreamLit web interface:

```
streamlit run inference_app/streamlit_app.py
```

* FastAPI inference engine:

```
python3 inference_app/fastapi_app.py
```
