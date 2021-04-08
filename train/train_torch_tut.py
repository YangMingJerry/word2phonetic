# -*- coding: utf-8 -*-
# Author    ：Yang Ming
# Create at ：2021/4/8
# tool      ：PyCharm

from converter.torch_tut_source import device,train,trainIters,prepareData
from model.torch_model_tut import EncoderRNN, AttnDecoderRNN


input_lang, output_lang, pairs = prepareData('eng', 'gb', False)
hidden_size = 20
encoder1 = EncoderRNN(input_lang.n_words+1, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words+1, dropout_p=0.1).to(device)
trainIters(encoder1, attn_decoder1, input_lang, output_lang, pairs, 100000, print_every=500)