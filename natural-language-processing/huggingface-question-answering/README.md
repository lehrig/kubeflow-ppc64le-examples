# Accelerate AI inference with IBM Power10 MMA and Kubeflow on Power
## Question Answering demo

For more information, please see the [related IBM TechZone asset](https://techzone.ibm.com/collection/ai-inference-mma-and-kubeflow).

---

## Folder content

* `inference_app/`: source code for the web application
	* `fastapi/`: source code and Dockerfile to build the FastAPI component (querying the AI backend)
	* `streamlit/`: source code and Dockerfile to build the StreamLit component (web frontend written in Python)
	* `app.yaml`: YAML file for deploying the web application in Openshift (creating a Deployment with 2 containers, a Service and a Route)
* `pipeline/`: Kubeflow pipeline code
	* `question-answering.ipynb`: Jupyter notebook for training, exporting and deploying the AI model
	* `deploy-mma.yaml` and `deploy-no-mma.yaml`: YAML files for deploying the MMA and non-MMA model serving containers (based on KServe)
* `training/train_huggingface_transformer.py`: Python source code of the training script used in the Kubeflow pipeline (allowing to generated the ONNX trained model without Kubeflow)

---

# AI model

TODO

---

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

---

# Kubeflow Pipeline

TODO

---

# Sources

* https://huggingface.co/docs/transformers/main/en/tasks/question_answering
* https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb

# Contacts

[Jeremie Chheang](mailto:jeremie.chheang@ibm.com)
MLOps Intern
IBM Client Engineering for Systems | EMEA
Montpellier, France

[Maxime Deloche](mailto:maxime.deloche1@ibm.com)
Deep Learning Engineer
IBM Client Engineering for Systems | EMEA
Montpellier, France

[Marvin Gießing](marving@de.ibm.com)
AI & Red Hat Openshift Systems Architect
IBM Technology Sales, DACH
Frankfurt, Germany
