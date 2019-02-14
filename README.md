# Under construction  

# YOLOv2 via Chainer to predict objects and apply its intel NCS

[Original README](./README_original.md)  

## Requirements

- python3 and pip3
- chainer 5.2.0( need up to 2.0.0 version )

## Environment for test

- ubuntu 16.04

## prepare
    $ pip3 install chainer==5.2.0
    
## Download and first move
    $ git clone https://github.com/k5iogura/YOLOv2-chainer
    $ cd YOLOv2-chainer
    $ cat model/yolo.weights_* > yolo.weights
    $ python3 yolov2_darknet_parser.py yolo.weights
    $ python3 yolov2_darknet_predict.py dog.jpg
    loading image...
    loading coco model...
    person(81%)
    person(65%)
    person(73%)
    person(74%)
    person(75%)
    save results to yolov2_result.jpg

![](./files/first_view.png)

### Notice to prevent UTF-8 coding
Some python scripts in this repo. are coded by UTF-8 to use japanese. It causes error of python runtime, so apply one of bellows.  

- Use Japanese Ubuntu
- Add *# encoding: utf-8* to top line of some python scripts(done)
- Use anaconda python

## Output memory layout of Inference engine of chainer
**Flow of yolov2_darknet_predict.py and memory layout of in/out predictor.**  

    orig_img.shape (576, 768, 3)             # read "dog.jpg" image by opencv
    reshaped to orig_img.shape (320, 448, 3) # transform to optimal size
    BGR2RGB                                  # change BGR to RGB
    img/255                                  # normalize 0.0 to 1.0
    transepose img.shape (3, 320, 448)       # transform HWC to CHW
    new axis .shape (1, 3, 320, 448)         # transform CHW to NCHW
    variable.shape (1, 3, 320, 448)          # create chainer varable
    predicted x.shape (1, 5, 1, 10, 14)      # X
    predicted y.shape (1, 5, 1, 10, 14)      # Y
    predicted w.shape (1, 5, 1, 10, 14)      # W
    predicted h.shape (1, 5, 1, 10, 14)      # H
    predicted conf.shape (1, 5, 1, 10, 14)   # Confidence
    predicted prob.shape (1, 5, 80, 10, 14)  # Class probabilities

### Transforming original image size to optimal size
Input image size of YOLO network will be transformed to optimal size divided by 32. 32 is downsampling ratio of YOLO(1/2^5).  

# Swaping inference engine from chainer to OpenVINO

## Try to transform from chainer to formats supported by OpenVINO

### to caffemodel

    Exception: Cannot convert, name=BroadcastTo-1-1, rank=1,
    label=BroadcastTo, inputs=['Reshape-0-1']

chainer.exporters does not know Reshape process in YOLOv2.  

### to onnx

first of all, install onnx-chainer

    $ pip3 install onnx-chainer
    ...
    Collecting chainer>=3.2.0 (from onnx-chainer)
    Collecting onnx>=1.3.0 (from onnx-chainer)
    ...
    onnx_op_name, opset_versions = mapping.operators[func_name]
    KeyError: 'BroadcastTo'

onnx_chainer output error.  

## Limitation to convert chainer network definition to onnx or caffemodel  

- **L.Bias causes KeyError: 'BroadcastTo'. So do not use L.Bias layer.**
- **use_beta option of L.BatchNormalize() must be True. So do not use use_beta=False**
- **Rewrite Network definition to pass above 2 items.**

## transform chainer .model to IRmodel .bin, .xml

OpenVINO IE outputs inference result as (1,83300) memory layout.  

    83300 = 14*14*5*85 = grids * girds * num * (classes + coords + conf)

