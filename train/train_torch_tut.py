# -*- coding: utf-8 -*-
# Author    ：Yang Ming
# Create at ：2021/4/8
# tool      ：PyCharm
import time
import torch

from converter.torch_tut_source import device,train,trainIters,prepareData
from model.torch_model_tut import EncoderRNN, AttnDecoderRNN

def save_model(encoder, decoder):
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    PATH_en = f'../model/encoder_{timestamp}'
    PATH_de = f'../model/decoder_{timestamp}'
    torch.save(encoder.state_dict(), PATH_en)
    torch.save(decoder.state_dict(), PATH_de)

def train_and_save(hidden_size,epochs_num, print_every):
    input_lang, output_lang, pairs = prepareData('eng', 'gb', False)
    encoder1 = EncoderRNN(input_lang.n_words+1, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words+1, dropout_p=0.1).to(device)
    trainIters(encoder1, attn_decoder1, input_lang, output_lang, pairs, epochs_num, print_every=print_every)
    save_model(encoder1,attn_decoder1)

if __name__ == '__main__':
    train_and_save(hidden_size=20,epochs_num=100000,print_every=500)