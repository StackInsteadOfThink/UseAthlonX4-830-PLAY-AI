# -*- coding: utf-8 -*-
"""冒烟测试: 真实300万模型跑60迭代, 验证训练+测速"""
import sys
sys.path.insert(0, r"D:\MiniChat Beta 0.1")
import train
train.CFG["max_iters"] = 60
train.CFG["ckpt_interval"] = 30
train.CFG["eval_interval"] = 20
train.CKPT = r"D:\MiniChat Beta 0.1\smoke.npz"
train.LOG = r"D:\MiniChat Beta 0.1\smoke_log.txt"
import os
if os.path.exists(train.CKPT): os.remove(train.CKPT)
if os.path.exists(train.LOG): os.remove(train.LOG)
train.train()
