# Under construction  

# YOLOv2 via Chainer to predict objects and apply its intel NCS

[Original README](./README_original.md)  

## Requirements

- python3 and pip3
- chainer 1.17.0

## Environment for test

- ubuntu 16.04

## prepare
    $ pip3 install chainer==1.17.0
    
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
- Add *# encoding: utf-8* to top line of some python scripts
- Use anaconda python
