#! /usr/bin/env python3
# encoding: utf-8
import sys, os
import argparse
import numpy as np
import chainer
import chainer.links as L
from chainer import serializers
from yolov2_orig import YOLOv2

parser = argparse.ArgumentParser(description="hdf5 to npz transform")
parser.add_argument('hdf5', help="input hdf5 format")
parser.add_argument('npz',  help="input npz format")
args = parser.parse_args()

print("define model (80,5)")
model = YOLOv2(80,5)

if os.path.exists(args.hdf5):
    print("load hdf5 model",args.hdf5)
    serializers.load_hdf5(args.hdf5, model)
else:
    print(args.hdf5,"not found")
    sys.exit(-1)

x = np.zeros((1, 3, 416, 416), dtype=np.float32)

chainer.config.train = False

with chainer.using_config('train',False):
    print("infer for dummy data", x.shape)
    result = model(x)
    print("result",result.shape, type(result))
    print("save as npz model", args.npz)
    serializers.save_npz(args.npz, model)
