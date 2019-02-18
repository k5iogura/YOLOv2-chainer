# Under construction  

# YOLOv2 via Chainer to predict objects and apply its intel NCS

[Original README](./README_original.md)  

## Requirements

- python3 and pip3
- chainer 5.2.0( need at least 2.0.0 version )

## Environment for test

- ubuntu 16.04

## prepare
    $ pip3 install chainer==5.2.0
    
## Download and first move
    $ git clone *this_repo*
    $ cd *this_repo* directory
    $ cat model/yolo.weights_* > yolo.weights
    $ python3 yolov2_darknet_parser.py yolo.weights
    loading yolo.weights
    loading initial model...
    1 992
    2 19680
    3 93920
    4 102368
    5 176608
    6 472544
    7 505824
    8 801760
    9 1983456
    10 2115552
    11 3297248
    12 3429344
    13 4611040
    14 9333728
    15 9860064
    16 14582752
    17 15109088
    18 19831776
    19 29273056
    20 38714336
    21 67029984
    22 67465609
    save weights file to yolov2_darknet.model
    
    $ python3 yolov2_darknet_predict.py dog.jpg 
    loading image...
    loading coco model...
    orig_img.shape (576, 768, 3)
    reshaped to orig_img.shape (320, 448, 3)
    BGR2RGB
    img/255
    transepose img.shape (3, 320, 448)
    new axis .shape (1, 3, 320, 448)
    variable.shape (1, 3, 320, 448)
    call self.model.predict
    predicted x.shape (1, 5, 1, 10, 14)
    predicted y.shape (1, 5, 1, 10, 14)
    predicted w.shape (1, 5, 1, 10, 14)
    predicted h.shape (1, 5, 1, 10, 14)
    predicted conf.shape (1, 5, 1, 10, 14)
    predicted prob.shape (1, 5, 80, 10, 14)
    car(76%)
    dog(66%)
    bicycle(59%)
    save results to yolov2_result.jpg

![](./files/first_view.png)

### Notice to prevent UTF-8 coding
If using original repo instead of this repo, some python scripts in original repo are coded by UTF-8 to use japanese. It causes error of python runtime, so apply one of bellows.  

- Use Japanese Ubuntu
- Add *# encoding: utf-8* to top line of some python scripts(done)
- Use anaconda python

## Investigate output memory layout of Inference engine of chainer
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

## Try to transform from chainer to other formats supported by OpenVINO

Notice: Transformation from Darknet weights and cfg to tensorflow pb is out of scope of this repo.  

### to caffemodel

    Exception: Cannot convert, name=BroadcastTo-1-1, rank=1,
    label=BroadcastTo, inputs=['Reshape-0-1']

ONNX does not know BroadcastTo operation made by L.Bias in YOLOv2.  

### to onnx

install onnx-chainer  

    $ pip3 install onnx-chainer
    ...
    Collecting chainer>=3.2.0 (from onnx-chainer)
    Collecting onnx>=1.3.0 (from onnx-chainer)
    ...
    onnx_op_name, opset_versions = mapping.operators[func_name]
    KeyError: 'BroadcastTo'

onnx_chainer output error.  

## Limitations to convert chainer network definition to onnx or caffemodel  

- **L.Bias causes KeyError: 'BroadcastTo'. So do not use L.Bias layer.**
- **use_beta option of L.BatchNormalize() must be True. So do not use use_beta=False**
- **Rewrite Network definition to keep 2 terms of above.**

## Modify chainer model definition avoid onnx-chainer troubles

Let you see modified version ./yolov2.py.  
And confirm the differences btn ./yolov2.py and ./yolov2_orig.py.  
Using ./yolov2.py as sample, you can continue bellow sections.  

## transform chainer .model to IRmodel .bin, .xml

OpenVINO IE outputs inference result as (1,83300) memory layout.  

    83300 = 14*14*5*85 = grids * girds * num * (classes + coords + conf)

