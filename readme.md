### Tensorflow Object Detection API on Webcam
---
This repository is a small implementation of object detection on webcam using tensorflow's object detection API.

### Getting Started
---
Before starting, we need to get the modules of Tensorflow object detection by running the shell script.
```shell
bash get-TF_OD_API.sh
```
From the ```main.py``` script, there are some directories  need to be changed for different needs.

* isDownload: Whether to download weights
* MODEL_NAME: The name of the model that will be used
* PATH_TO_FROZEN_GRAPH: The path to the .pb
* PATH_TO_LABELS: The path to .pbtxt label file

The script uses **SSD** and **Mobilenet_v1**. Tensorflow provides multiple pre-trained models which can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

After change the directories related parameters, run the script below for object detection on webcam.
```shell
python main.py
```
This [article](https://medium.com/guan-hong/tensorflow-object-detection-api-de6322cbc500) contains details for running tensorflow object detectin API.
### Acknowledgment
---
1. [Tensorflow Object Detection API](https://github.com/tensorflow/models)
2. [Google's Object Detectin API Tutorial](https://blog.gtwang.org/programming/tensorflow-object-detection-api-tutorial/)

