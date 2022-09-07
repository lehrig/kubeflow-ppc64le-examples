# How to run a Katib experiment with Kubeflow Pipelines

Katib is a framework native to Kubernetes that also works with RedHat Openshift. Its purpose is to tune hyperparameters. \
The idea is to create a YAML file that contains:
- The parameters of the model
- The number of trials in total
- The number of parallel trials
- The number of failed trials allowed
- An objective function
- A search algorithm
- A metric collector specs that indicates how metrics are collected. The default option is StdOut but in this example, we collect metrics in a file. The default format is "metric_name=value". 
- A trial template that contains necessary information for running the experiment, including the container specs and the image that contains the model to tune, a command to launch and a set of parameters to tune. In this example, GPU usage has also been set in the "resources" section of the trial spec JSON. 

The final yaml looks like the **example.yaml** file. However, if you want to run Katib using Kubeflow Pipelines from a Jupyter notebook, you must create the previous yaml file using Python and the Kubeflow-Katib SDK. The SDK is compatible with Kubeflow v1beta1 and can be installed using <code>pip install kubeflow-katib</code>. \
The SDK contains all the classes for creating the YAML file requirements listed above. The documentation can be found in the following Github repo: https://github.com/kubeflow/katib/tree/master/sdk/python/v1beta1/docs

Finally, the experiment component is created from Kubeflow launcher component. This one has been rebuilt for IBM Power Systems and has to be loaded from the file **katib-pipelines-launcher.yaml** in this folder. \

The pipeline graph looks like this: \

![Katib Pipeline](../images/katib_pipeline.png "Katib Pipeline")