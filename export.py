# encoding: utf-8
import numpy as np
import chainer
# from chainer.exporters import caffe
from chainer import serializers
import chainer.links as L
import onnx_chainer
import onnx
#import cupy
from yolov2 import YOLOv2

npz_weight_file  = 'yolov2_darknetNoBias.npz'
onnx_weight_file = 'yolov2_darknetNoBias.onnx'

model = YOLOv2(80,5)
serializers.load_npz(npz_weight_file,model)

x = np.zeros((1, 3, 416, 416), dtype=np.float32)

chainer.config.train = False

with chainer.using_config('train',False):
#     print("save as caffemodel")
#     caffe.export(model, [chainer.Variable(x)], None, True,'test')
    print("save as onnx model")
    onnx_model = onnx_chainer.export(model, x, filename=onnx_weight_file, save_text=True)
    print("try to load onnx model in NoBias-YOLOv2")
    modelx = onnx.load(onnx_weight_file)
    print("onnx_model quickly check for onnx nodes=",len(modelx.graph.node))
    for i , node in enumerate(modelx.graph.node):
        print("[Node #{}]".format(i))
        print(node)
        if i==2: break
    print("at least 2 nodes printed out")
