from __future__ import print_function
import numpy as np
import pickle as pkl
import sys
import argparse
from wordvec_model import WordVec
from glove_model import GloveVec
from rnnvec_model import RnnVec
from prepare_file import prepare_file

def find_max_length(file_name):
    temp_len = 0
    max_length = 0
    for line in open(file_name):
        if line in ['\n', '\r\n']:
            if temp_len > max_length:
                max_length = temp_len
            temp_len = 0
        else:
            temp_len += 1
    return max_length

def pos(tag):
    one_hot = np.zeros(5)
    if tag == 'NN' or tag == 'NNS':
        one_hot[0] = 1
    elif tag == 'FW':
        one_hot[1] = 1
    elif tag == 'NNP' or tag == 'NNPS':
        one_hot[2] = 1
    elif 'VB' in tag:
        one_hot[3] = 1
    else:
        one_hot[4] = 1

    return one_hot

def chunk(tag):
    one_hot = np.zeros(5)

    if 'NP' in tag:
        one_hot[0] = 1
    elif 'VP' in tag:
        one_hot[1] = 1
    elif 'PP' in tag:
        one_hot[2] = 1
    elif tag == 'O':
        one_hot[3] = 1
    else:
        one_hot[4] = 1

    return one_hot

def capital(word):
    if ord('A') <= ord(word[0]) <= ord('Z'):
        return np.array([1])
    else:
        return np.array([0])

def get_input(model, word_dim, input_file, sentence_length=-1):
    print('processing %s' % input_file)
    word = []
    tag = []
    sentence = []
    sentence_tag = []

    if sentence_length == -1:
        max_sentence_length = find_max_length(input_file)
    else:
        max_sentence_length = sentence_length

    sentence_length = 0

    print("max sentence length is %d" % max_sentence_length)

    for line in open(input_file):
        if line in ['\n', '\r\n']:
            for _ in range(max_sentence_length - sentence_length):
                tag.append(np.array([0] * 10))
                temp = np.array([0 for _ in range(word_dim + 11)])
                word.append(temp)
            sentence.append(word)
            sentence_tag.append(np.array(tag))
            sentence_length = 0
            word = []
            tag = []

        else:
            assert (len(line.split()) == 4)
            sentence_length += 1
            temp = model[line.split()[0]]
            assert len(temp) == word_dim
            temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
            temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
            temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
            word.append(temp)

    assert (len(sentence) == len(sentence_tag))
    return sentence 

def convert_file(file):
    word_dim = 311
    input_file = file

    output_array = get_input(model_glove, word_dim, input_file)
    return output_array
