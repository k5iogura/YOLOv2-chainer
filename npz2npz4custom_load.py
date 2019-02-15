#! /usr/bin/env python3
# encoding: utf-8
import sys, os
import argparse
import numpy as np
import chainer
import chainer.links as L
from chainer import serializers
from yolov2 import YOLOv2, load_npz

parser = argparse.ArgumentParser(description="npz1 to npz transform")
parser.add_argument('npz1', help="input npz format")
parser.add_argument('npz2', help="output npz format")
parser.add_argument('--model','-m', type=str, default="yolov2",help="target model file")
args = parser.parse_args()

model_prefix,_ = os.path.splitext(os.path.basename(args.model))
print("from",model_prefix,"import YOLOv2")
exec("from "+model_prefix+" import YOLOv2")

print("define model (80,5)")
model = YOLOv2(80,5)

if os.path.exists(args.npz1):
    print("custom load npz1 model",args.npz1)
    load_npz(args.npz1, model)
else:
    print(args.npz1,"not found")
    sys.exit(-1)

x = np.zeros((1, 3, 416, 416), dtype=np.float32)

chainer.config.train = False

with chainer.using_config('train',False):
    print("infer for dummy data", x.shape)
    result = model(x)
    print("result",result.shape, type(result))
    print("save as npz model", args.npz2)
    serializers.save_npz(args.npz2, model)

    print("serializers.load_npz model",args.npz2)
    serializers.load_npz(args.npz2, model)
    print("infer No.2 for dummy data", x.shape)
    result = model(x)
    print("result No.2",result.shape, type(result))

