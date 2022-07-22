# Accelerate AI inference with IBM Power10 MMA and Kubeflow on Power
## Question Answering demo

For more information, please see the [related IBM TechZone asset](https://techzone.ibm.com/collection/ai-inference-mma-and-kubeflow).


## Folder content

* `inference_app/`: source code for the web application
	* `fastapi/`: source code and Dockerfile to build the FastAPI component (querying the AI backend)
	* `streamlit/`: source code and Dockerfile to build the StreamLit component (web frontend written in Python)
	* `app.yaml`: YAML file for deploying the web application in Openshift (creating a Deployment with 2 containers, a Service and a Route)
* `pipeline/`: Kubeflow pipeline code
	* `question-answering.ipynb`: Jupyter notebook for training, exporting and deploying the AI model
	* `deploy-mma.yaml` and `deploy-no-mma.yaml`: YAML files for deploying the MMA and non-MMA model serving containers (based on KServe)
* `training/train_huggingface_transformer.py`: Python source code of the training script used in the Kubeflow pipeline (allowing to generated the ONNX trained model without Kubeflow)


# AI model

The Deep Learning model used is a [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) Transformer model from the [HuggingFace](https://huggingface.co/docs/transformers/index) library.

We fine-tune it for a question-answering use case on the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/).
This type of task means that the model takes a "question" and a "context" as input, and returns the answer of the question in the context.

The source code used is inspired by:

* https://huggingface.co/docs/transformers/main/en/tasks/question_answering
* https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb

In this demo, we want the user to provide the question only. We implemented a "context retrieval" part that first use [spaCy](https://spacy.io/) to extract the entity in the question, then the [Wikipedia API](https://pypi.org/project/wikipedia/) and Python wrapper to fetch a summary associated with that entity.

# Model deployment (without training)

## Prerequisites:

- Power10 LPAR
- Based on the Option you need:
    - A container engine (e.g. Docker/Podman)
    - A Kubernetes or OpenShift environment
    - A Kubeflow Installation

## Option 1: Using containers

### 1.1: Login to your P10 LPAR  & prepare the model folder structure and download the pretrained model:

```bash
mkdir -p ${HOME}/model_repository/onnx/question_answering/1
wget https://ibm.box.com/shared/static/mfdcjzx8tdkpyy7mrfzyoryev4g0g5s2.onnx -O ${HOME}/model_repository/onnx/question_answering/1/model.onnx
```

### 1.2: Deploy the container

```bash
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -d --name question-answering -v ${HOME}/model_repository/onnx:/models quay.io/mgiessing/tritonserver:22.06-py3-mma tritonserver --model-repository=/models --strict-model-config=false
```

Optionally check logs with `docker logs question-answering`

### 1.3: Health check

```bash
curl localhost:8000/v2
{"name":"triton","version":"2.23.0","extensions":["classification","sequence","model_repository","model_repository(unload_dependents)","schedule_policy","model_configuration","system_shared_memory","cuda_shared_memory","binary_tensor_data","statistics","trace"]}

curl localhost:8000/v2/models/question_answering
{"name":"question_answering","versions":["1"],"platform":"onnxruntime_onnx","inputs":[{"name":"attention_mask","datatype":"INT64","shape":[-1,-1]},{"name":"input_ids","datatype":"INT64","shape":[-1,-1]}],"outputs":[{"name":"end_logits","datatype":"FP32","shape":[-1,-1]},{"name":"start_logits","datatype":"FP32","shape":[-1,-1]}]}
```

## Option 2: Using Kserve standalone in K8s/OCP

TODO

## Option 3: Using Kserve in Kubeflow

### Get model into PVC

Prepare PV/PVC following this instructions: https://kserve.github.io/website/modelserving/storage/pvc/pvc/#create-pv-and-pvc

Copy the model into the PVC following this instructions: https://kserve.github.io/website/modelserving/storage/pvc/pvc/#copy-model-to-pv

Go the the Kubeflow UI and click on models -> New model server

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "question-answering"
  annotations:
    sidecar.istio.io/inject: "false"
spec:
  predictor:
    nodeSelector:
      mma: "true"
    containers:
      - name: triton
        image: quay.io/mgiessing/tritonserver:22.06-py3-nomma
        args:
        - tritonserver
        - --model-store=/mnt/models
        - --http-port=8080
        - --allow-http=true
        - --strict-model-config=false
        env:
          - name: STORAGE_URI
            value: "pvc://models-volume/models"
        resources:
          limits:
            cpu: "8"
            memory: 16Gi
          requests:
            cpu: "2"
            memory: 8Gi
```

# Web application

## Overview

The web interface is made of 2 components:

* a "streamlit" component for the actual web frontend (based on the [StreamLit](http://streamlit.io/) library)
* a "fastapi" component that receives frontend requests, handles interaction with the deployed models and return the results (based on the [FastAPI](https://fastapi.tiangolo.com/) library)

For each, we built images available at the following URLs:

* [quay.io/mdeloche/kubeflow-streamlit](https://quay.io/repository/mdeloche/kubeflow-streamlit)
* [quay.io/mdeloche/kubeflow-fastapi](https://quay.io/repository/mdeloche/kubeflow-fastapi)

## How to deploy

To deploy the web application on Openshift:

1. Make sure you have the Openshift CLI `oc` installed and that you are connected to your Openshift cluster.

2. Open the `inference_app/app.yaml` file. In the `kind: Route` definition, update the `spec.host` field to match your Openshift cluster host name.

3. Create the components using:

```
oc apply -f inference_app/app.yaml
```

4. Navigate to the URL that you defined in the `app.yaml` file: the StreamLit interface should be displayed here.


## How to use the interface

The first text field allows to input the URL of your deployed model (see instructions below). It should end with `/v2/models/question-answering/infer` unless you changed it.

You can optionally use the **Check backend status** button to send a dummy request to the model and verify that it is successfully reached (in that case, a green checkmark will appear).

You can then use it: either pick a sample question in the right column, or input your own, then click **Submit**.
The app should answer, and the inference time will be displayed in the left column.

> Note 1: the app is designed to search for an entity in the question, try to fetch its associated Wikipedia summary and give both as input to the AI model. You need to make sure there is an "obvious" entity in the question in order for it to work.

> Note 2: The app allows to select multiple backends in the left bar. Currently, only the Nvidia Triton Inference Server has been implemented and tested.


## How to build images (optional)

To build the images using a local Dockerfile and Openshift, you can use the following commands (replacing your username, password, and quay.io username):

```
oc create secret docker-registry quay.io --docker-username <XXXXXXX> --docker-password <XXXXXX> --docker-server https://quay.io/<XXXXXX>

oc new-build --binary=true --name=<myapp> --push-secret=quay.io --to-docker=true --to quay.io/<XXXXXX>/image:tag 
```

Then, you can launch a new build and automatically push the image with:

```
oc start-build <myapp> --from-dir=. --follow
```


# Kubeflow Pipeline

TODO


# Contacts

[Jeremie Chheang](mailto:jeremie.chheang@ibm.com)<br />
MLOps Intern<br />
IBM Client Engineering for Systems | EMEA<br />
Montpellier, France

[Maxime Deloche](mailto:maxime.deloche1@ibm.com)<br />
Deep Learning Engineer<br />
IBM Client Engineering for Systems | EMEA<br />
Montpellier, France

[Marvin Gießing](marving@de.ibm.com)<br />
AI & Red Hat Openshift Systems Architect<br />
IBM Technology Sales, DACH<br />
Frankfurt, Germany
