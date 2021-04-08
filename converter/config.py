# -*- coding: utf-8 -*-
# Author    ：Yang Ming
# Create at ：2021/4/8
# tool      ：PyCharm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 128
teacher_forcing_ratio = 0.5
data_path = '../data/gb_data.train'