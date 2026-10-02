# -*- coding: utf-8 -*-
"""直接读取 beat / downbeat 模型 pickle 的真实输入维，并搜索能产出 162 的特征配置。"""
import os, sys, types, glob
import numpy as np
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                   ".madmom_pkg", "madmom-0.16.1", "madmom")

import collections, collections.abc
for _n in ("MutableSequence","MutableMapping","MutableSet","Sequence","Mapping",
           "Set","Iterable","Iterator","Callable","Hashable","ABCMeta"):
    if not hasattr(collections, _n) and hasattr(collections.abc, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

for name, mod in [("madmom", SRC), ("madmom.ml", os.path.join(SRC,"ml")),
                  ("madmom.audio", os.path.join(SRC,"audio"))]:
    if name not in sys.modules:
        mm = types.ModuleType(name); mm.__path__ = [mod]; sys.modules[name] = mm
if "distutils" not in sys.modules:
    du = types.ModuleType("distutils"); duv = types.ModuleType("distutils.version")
    class LV:
        def __init__(s,v): s.v=str(v)
        def _c(s,o):
            import re
            a=[int(x) for x in re.findall(r"\d+",s.v)]; b=[int(x) for x in re.findall(r"\d+",str(o))]
            return (a>b)-(a<b)
        def __lt__(s,o): return s._c(o)<0
        def __gt__(s,o): return s._c(o)>0
        def __eq__(s,o): return s._c(o)==0
    duv.LooseVersion=LV; du.version=duv; sys.modules["distutils"]=du; sys.modules["distutils.version"]=duv

MODELS = os.path.join(SRC, "models")
def load(path):
    from madmom.ml.nn import NeuralNetwork
    with open(path,"rb") as f:
        return pickle.load(f, encoding="latin-1") if False else __import__("pickle").load(f, encoding="latin-1")

import pickle
def input_dim(nn):
    for layer in nn.layers:
        cls = type(layer).__name__
        if cls in ("LSTMLayer","GRULayer"):
            g = layer.input_gate if hasattr(layer,"input_gate") else layer.reset_gate
            return int(np.asarray(g.weights).shape[0])
        if hasattr(layer,"fwd_layer"):
            g = layer.fwd_layer.input_gate if hasattr(layer.fwd_layer,"input_gate") else layer.fwd_layer.reset_gate
            return int(np.asarray(g.weights).shape[0])
        if cls=="FeedForwardLayer":
            return int(np.asarray(layer.weights).shape[0])
    return None

for pat,label in [("beats/2015/beats_blstm_1.pkl","beat BLSTM 2015"),
                  ("beats/2016/beats_lstm_1.pkl","beat LSTM 2016"),
                  ("downbeats/2016/downbeats_blstm_1.pkl","downbeat BLSTM 2016")]:
    p = os.path.join(MODELS, pat)
    if os.path.isfile(p):
        nn = load(p)
        print("%-22s input_dim=%s  out_dim=%d" % (label, input_dim(nn), nn.layers[-1].weights.shape[1] if hasattr(nn.layers[-1],'weights') else '?'))

# ---- 搜索能产出 162（81 带）的配置 ----
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("vb", os.path.join(os.path.dirname(__file__),"verify_bands.py"))
# 直接复用 LogarithmicFilterbank 已 import
from madmom.audio.filters import LogarithmicFilterbank
SR=44100
def bands(fs,bpo):
    nbins=fs//2
    bf=np.arange(nbins,dtype=np.float64)*(SR/fs)
    fb=LogarithmicFilterbank(bf,num_bands=bpo,fmin=30.0,fmax=17000.0,fref=440.0,norm_filters=True,unique_filters=True)
    return fb.shape[1]
print("\n--- 单分辨率 (frame_size,bpo)->带数 ---")
for fs in (512,1024,2048,4096):
    row=[]
    for bpo in (3,6,12,24):
        row.append("%d@%d=%d"%(fs,bpo,bands(fs,bpo)))
    print("  ".join(row))
print("\n目标: 总带数=81 ( => 162 特征 ). 3 分辨率候选:")
fs_list=[512,1024,2048,4096]; bpo_list=[3,6,12,24]
best=[]
for a in fs_list:
    for b in fs_list:
        for c in fs_list:
            for ba in bpo_list:
                for bb in bpo_list:
                    for bc in bpo_list:
                        if bands(a,ba)+bands(b,bb)+bands(c,bc)==81:
                            best.append(((a,ba),(b,bb),(c,bc)))
for x in best[:20]:
    print("  ", x, "->", [bands(f,b) for f,b in x])
