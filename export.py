# encoding: utf-8
import numpy as np
import chainer
import chainer.links as L
import onnx_chainer
import onnx
#import cupy
from yolov2 import YOLOv2

#model = L.VGG16Layers()
model = YOLOv2(80,5)

# ネットワークに流し込む擬似的なデータを用意する
x = np.zeros((1, 3, 416, 416), dtype=np.float32)

# 推論モードにする
chainer.config.train = False

print("save as onnx model")
with chainer.using_config('train',False):
    onnx_model = onnx_chainer.export(model, x, filename='YOLOv2.onnx', save_text=True)
    print("load onnx model")
    modelx = onnx.load("YOLOv2.onnx")
    print("nodes",len(modelx.graph.node))
    for i , node in enumerate(modelx.graph.node):
        print("[Node #{}]".format(i))
        print(node)
        if i==15: break
