# -*- coding: utf-8 -*-
# Author    ：Yang Ming
# Create at ：2021/4/8
# tool      ：PyCharm
import torch

from converter.torch_tut_source import device,train,trainIters,prepareLang,evaluate
from seq2seq_exp.data_handler import DataHandler
from model.torch_model_tut import EncoderRNN, AttnDecoderRNN

from converter.config import *
data_handler = DataHandler()

path_en = '/home/ming/PycharmProjects/g2x_exp/model/encoder_2021-04-08-15-36-58'
path_de = '/home/ming/PycharmProjects/g2x_exp/model/decoder_2021-04-08-15-36-58'


def load_model_locally(path_en,path_de):
    input_lang, output_lang = prepareLang('eng', 'gb', False)
    hidden_size = 20
    encoder = EncoderRNN(input_lang.n_words+1, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words+1, dropout_p=0.1).to(device)

    encoder.load_state_dict(torch.load(path_en))
    attn_decoder.load_state_dict(torch.load(path_de))
    return encoder, attn_decoder

def convert(word):
    encoder, attn_decoder = load_model_locally(path_en,path_de)
    input_lang, output_lang = prepareLang('eng', 'gb', False)
    return ''.join(evaluate(encoder=encoder,decoder=attn_decoder,input_lang=input_lang,output_lang=output_lang,word=word,max_length=MAX_LENGTH)[0])

if __name__ == '__main__':
    word = 'decent'
    phonetic = convert(word)
    print(phonetic)


