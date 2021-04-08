# -*- coding: utf-8 -*-
# Author    ：Yang Ming
# Create at ：2021/4/8
# tool      ：PyCharm

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import time
import random
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch import optim
from seq2seq_exp.data_handler import DataHandler
from seq2seq_exp.UI import print_percent
from converter.config import *

time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2count = {}
        data_handler = DataHandler()
        if name == 'eng':
            self.word2index = data_handler.alpha2index
            self.index2word = data_handler.index2alpha
            self.n_words = data_handler.alphabet_count  # Count SOS and EOS
        else:
            self.word2index = data_handler.phonetic2index
            self.index2word = data_handler.index2phonetic
            self.n_words = data_handler.phonetic_count

    def addSentence(self, input):
        pass

def save_model(encoder, decoder, path_en, path_de):
    torch.save(encoder.state_dict(), path_en)
    torch.save(decoder.state_dict(), path_de)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(data_path, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0]) < MAX_LENGTH and \
        len(p[1]) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def prepareLang(lang1, lang2, reverse=False):
    if reverse:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang

def indexesFromSentence(lang, word):
    data_handler = DataHandler()
    # return [lang.alpha2index[s] for s in list(word)]
    if lang.name == 'eng':
        return [lang.word2index[s] for s in list(word)]
    else:
        return data_handler.dump_to_index_by_phonetic(word)

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder,input_lang , output_lang, pairs, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, save_during_train = False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    path_en = f'../model/best_encoder_{time_stamp}'
    path_de = f'../model/best_decoder_{time_stamp}'
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # pairs_chosen = [random.choice(pairs) for i in range(n_iters)]
    # training_pairs = [tensorsFromPair(input_lang , output_lang, pairs_chosen[i])
    #                   for i in range(n_iters)]
    criterion = nn.NLLLoss()
    min_loss = 100
    epoch_saved = 0
    for iter in range(1, n_iters + 1):
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0
        for i,pair in enumerate(pairs):
            training_pair = tensorsFromPair(input_lang , output_lang, pair)
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
            print_percent(i,len(pairs))

        print_loss_avg = print_loss_total / len(pairs)
        print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                     iter, iter / n_iters * 100, print_loss_avg))
        plot_loss_avg = print_loss_total
        plot_losses.append(plot_loss_avg)
        if print_loss_avg < min_loss and save_during_train:
            #save
            save_model(encoder,decoder,path_en,path_de)
            epoch_saved = iter
        showPlot(plot_losses)
    print(f'model of epoch {epoch_saved} is saved ')

def showPlot(points):
    plt.figure()
    plt.subplots()
    plt.plot(points)
    plt.savefig(f'losses{time_stamp}.png')
    plt.show()

def evaluate(encoder, decoder, input_lang, output_lang, word, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, word)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, input_lang , output_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder,input_lang , output_lang, pair[0])
        output_sentence = ''.join(output_words)
        print('<', output_sentence)
        print('')


