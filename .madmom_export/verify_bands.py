# -*- coding: utf-8 -*-
"""用 madmom 真实的 LogarithmicFilterbank 计算各分辨率的真实带数，
确认 Features.cs 的 2*(b0+b1+b2) 恰好等于模型输入维 314 / 162。"""
import os
import sys
import types
import numpy as np
# madmom 0.16.1 用了已移除的 np.float 别名
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                   ".madmom_pkg", "madmom-0.16.1", "madmom")

# ---- 轻量 stub（仅 Processor / integer_types），filters.py 只依赖这些 ----
import collections, collections.abc
for _n in ("MutableSequence", "MutableMapping", "MutableSet", "Sequence",
           "Mapping", "Set", "Iterable", "Iterator", "Callable", "Hashable", "ABCMeta"):
    if not hasattr(collections, _n) and hasattr(collections.abc, _n):
        setattr(collections, _n, getattr(collections.abc, _n))

if "madmom" not in sys.modules:
    m = types.ModuleType("madmom"); m.__path__ = [SRC]
    sys.modules["madmom"] = m
if "madmom.audio" not in sys.modules:
    ma = types.ModuleType("madmom.audio"); ma.__path__ = [os.path.join(SRC, "audio")]
    sys.modules["madmom.audio"] = ma
if "madmom.processors" not in sys.modules:
    mp = types.ModuleType("madmom.processors")
    class Processor:
        pass
    class BufferProcessor(Processor):
        pass
    class SequentialProcessor(Processor):
        pass
    mp.Processor = Processor; mp.BufferProcessor = BufferProcessor
    mp.SequentialProcessor = SequentialProcessor
    sys.modules["madmom.processors"] = mp
if "madmom.utils" not in sys.modules:
    mu = types.ModuleType("madmom.utils")
    mu.integer_types = (int,)
    sys.modules["madmom.utils"] = mu

from madmom.audio.filters import LogarithmicFilterbank

SR = 44100
FREF = 440.0

def count_bands(frame_size, bpo):
    nbins = frame_size // 2
    bin_freq = np.arange(nbins, dtype=np.float64) * (SR / frame_size)
    fb = LogarithmicFilterbank(bin_freq, num_bands=bpo, fmin=30.0, fmax=17000.0,
                               fref=FREF, norm_filters=True, unique_filters=True)
    return fb.shape[1]  # (num_bins, num_bands) → num_bands

def report(name, config):
    bands = [count_bands(fs, bpo) for fs, bpo in config]
    total = 2 * sum(bands)
    print("%-10s 各分辨率带数=%s  2*sum=%d" % (name, bands, total))
    return total

db = report("downbeat", [(1024, 3), (2048, 6), (4096, 12)])
bt = report("beat", [(1024, 6), (2048, 6), (4096, 6)])
print("downbeat 目标 314 ->", "OK" if db == 314 else "MISMATCH")
print("beat     目标 162 ->", "OK" if bt == 162 else "MISMATCH")
