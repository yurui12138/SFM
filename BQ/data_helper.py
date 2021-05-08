# -*- coding: utf-8 -*-

import re
import pickle
import numpy as np
from gensim import models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def read_data(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = f.readlines()
    data = [re.split('\t\t', i) for i in data]
    q1 = [i[0] for i in data]
    q2 = [i[1] for i in data]
    label = [int(i[2]) for i in data]
    return q1, q2, label


# train_q1, train_q2, train_label = read_data('./data/cut_word/BQ_train1.txt')
# test_q1, test_q2, test_label = read_data('./data/cut_word/BQ_test1.txt')
# dev_q1, dev_q2, dev_label = read_data('./data/cut_word/BQ_dev1.txt')

train_q1, train_q2, train_label = read_data('./data/break_word/BQ_train1.txt')
test_q1, test_q2, test_label = read_data('./data/break_word/BQ_test1.txt')
dev_q1, dev_q2, dev_label = read_data('./data/break_word/BQ_dev1.txt')

corpus = train_q1 + train_q2 + test_q1 + test_q2 + dev_q1 + dev_q2
w2v_corpus = [i.split() for i in corpus]
word_set = set(' '.join(corpus).split())

vocab_size = len(word_set)

MAX_SEQUENCE_LENGTH = 30
EMB_DIM = 400

w2v_model = models.Word2Vec(w2v_corpus, size=EMB_DIM, window=5, min_count=1, sg=1, workers=4, seed=1234, iter=25)
w2v_model.save('w2v_model.pkl')
tokenizer = Tokenizer(num_words=len(word_set))
tokenizer.fit_on_texts(corpus)

train_q1 = tokenizer.texts_to_sequences(train_q1)
train_q2 = tokenizer.texts_to_sequences(train_q2)

test_q1 = tokenizer.texts_to_sequences(test_q1)
test_q2 = tokenizer.texts_to_sequences(test_q2)

dev_q1 = tokenizer.texts_to_sequences(dev_q1)
dev_q2 = tokenizer.texts_to_sequences(dev_q2)


train_pad_q1 = pad_sequences(train_q1, maxlen=MAX_SEQUENCE_LENGTH)
train_pad_q2 = pad_sequences(train_q2, maxlen=MAX_SEQUENCE_LENGTH)

test_pad_q1 = pad_sequences(test_q1, maxlen=MAX_SEQUENCE_LENGTH)
test_pad_q2 = pad_sequences(test_q2, maxlen=MAX_SEQUENCE_LENGTH)

dev_pad_q1 = pad_sequences(dev_q1, maxlen=MAX_SEQUENCE_LENGTH)
dev_pad_q2 = pad_sequences(dev_q2, maxlen=MAX_SEQUENCE_LENGTH)

embedding_matrix = np.zeros([len(tokenizer.word_index) + 1, EMB_DIM])
# np.random.seed(1024)
# embedding_matrix = np.random.random([len(tokenizer.word_index) + 1, EMB_DIM])


for word, idx in tokenizer.word_index.items():
    try:
        embedding_matrix[idx, :] = w2v_model.wv[word]
    except:
        print('1')

def save_pickle(fileobj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(fileobj, f)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        fileobj = pickle.load(f)
    return fileobj


model_data = {'train_q1': train_pad_q1, 'train_q2': train_pad_q2, 'train_label': train_label,
              'test_q1': test_pad_q1, 'test_q2': test_pad_q2, 'test_label': test_label,
              'dev_q1': dev_pad_q1, 'dev_q2': dev_pad_q2, 'dev_label': dev_label}

save_pickle(corpus, 'corpus.pkl')
save_pickle(model_data, 'model_data.pkl')
save_pickle(embedding_matrix, 'embedding_matrix.pkl')
save_pickle(tokenizer, 'tokenizer.pkl')
