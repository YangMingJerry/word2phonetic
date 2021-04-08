# -*- coding:utf-8 -*-
# CREATED BY: jiangbohuai
# CREATED ON: 2021/3/22 2:52 PM
# LAST MODIFIED ON:
# AIM:
from typing import List
import re
import numpy as np


class DataHandler:

    def __init__(self, data_path: str = ''):
        if data_path:
            with open(data_path, 'r') as f:
                self.data = f.read().splitlines()

        self.alphabet_map = {c: i+2 for i, c in enumerate('abcdefghijklmnopqrstuvwsxyz')}
        self.alphabet_map[0] = 'SOS'
        self.alphabet_map[1] = 'EOS'
        self.phonetic_alphabet = 'ɪə|eɪ|eə|əʊ|ʊə|ɔɪ|aɪ|aʊ|oʊ|ɪ|æ|e|ɜː|ə|iː|i|ʌ|uː|u|ʊ|ɔː|ɒ|ɑː|tʃ|tr|ts|dʒ|dr|dz|p|t|k|d|b|ɡ|f|s|ʃ|θ|h|v|z|ʒ|ð|r|m|n|ŋ|l|j|w|ˈp|ˈt|ˈk|ˈd|ˈb|ˈɡ|ˈf|ˈs|ˈʃ|ˈθ|ˈh|ˈv|ˈz|ˈʒ|ˈð|ˈr|ˈm|ˈn|ˈŋ|ˈl|ˈj|ˈw|ˌp|ˌt|ˌk|ˌd|ˌb|ˌɡ|ˌf|ˌs|ˌʃ|ˌθ|ˌh|ˌv|ˌz|ˌʒ|ˌð|ˌr|ˌm|ˌn|ˌŋ|ˌl|ˌj|ˌw|ˈɪə|ˈeɪ|ˈeə|ˈəʊ|ˈʊə|ˈɔɪ|ˈaɪ|ˈaʊ|ˈoʊ|ˈɪ|ˈæ|ˈe|ˈə|ˈi|ˈʌ|ˈu|ˈʊ|ˈɒ|ˌɪə|ˌeɪ|ˌeə|ˌəʊ|ˌʊə|ˌɔɪ|ˌaɪ|ˌaʊ|ˌoʊ|ˌɪ|ˌæ|ˌe|ˌə|ˌi|ˌʌ|ˌu|ˌʊ|ˌɒ'
        self.phonetic_re = re.compile(
            f"({self.phonetic_alphabet})")

        self.phonetic_math = {c: i+2 for i, c in enumerate(self.phonetic_alphabet.split('|'))}
        self.phonetic_math[0] = 'SOS'
        self.phonetic_math[1] = 'EOS'

        self.phonetic_count = 0
        self.alphabet_count = 0

        self.alpha2index = self.alphabet_map
        self.index2alpha = {}
        for key, value in self.alpha2index.items():
            self.index2alpha[value] = key
            self.alphabet_count += 1
        self.phonetic2index = self.phonetic_math
        self.index2phonetic = {}
        for key, value in self.phonetic2index.items():
            self.index2phonetic[value] = key
            self.phonetic_count += 1



    def to_one_hot(self, value: List[int], rng: int) -> np.array:
        length = len(value)
        out = np.zeros([length, rng])
        for i, v in enumerate(value):
            out[i][v] = 1
        return out

    def numeric_alphbeta(self, value: str):
        out = []
        for char in value:
            out.append(self.alphabet_map.get(char, 27))
        return out

    def numeric_phonetic(self, value: str):
        out = []
        rng = len(self.phonetic_alphabet.split('|')) + 1
        for char in self.phonetic_re.split(value):
            char = char.strip()
            if not char:
                continue
            try:
                out.append(self.phonetic_math[char])  # .get(char, rng))
            except:
                continue
                #print(char)
        return out

    def dump_to_numerical(self):
        for value in self.data:
            word, phonetic = value.split('\t')
            print(word, phonetic)
            word = self.numeric_alphbeta(word.lower())
            phonetic = self.numeric_phonetic(phonetic)
            yield word, phonetic

    def dump_to_index_by_phonetic(self, phonetic):
        return self.numeric_phonetic(phonetic)

    def dump_to_index_by_wordd(self, word):
        return self.numeric_alphbeta(word.lower())

    def index_to_word(self, word):
        return

if __name__ == '__main__':
    path = '../data/gb_data.train'
    handler = DataHandler(path)
    for word, phonetic in handler.dump_to_numerical():
        print(f'word=[{word}],phonetic=[{phonetic}]')
