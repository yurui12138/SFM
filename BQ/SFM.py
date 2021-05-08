# -*- coding: utf-8 -*-
import pickle
import sys
import time

import nni
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.backend import stack
from keras.engine import Layer
from keras.layers import Concatenate, Add, Flatten, Multiply, Conv3D, GRU, MaxPooling2D, AveragePooling2D
import keras
from keras.utils import plot_model

import data_helper_word

os.environ["CUDA_VISIBLE_DEVICES"] = "0,7"

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

KTF.set_session(sess)
from keras import backend as K, backend
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, Input, Bidirectional, Lambda, LSTM, \
    Reshape, Dense, Activation, concatenate, BatchNormalization, Conv2D
from keras.models import Model
from keras.optimizers import RMSprop
import data_helper

input_dim = data_helper.MAX_SEQUENCE_LENGTH
emb_dim = data_helper.EMB_DIM
model_path = './model/siameselstm.hdf5'
tensorboard_path = './model/ensembling'


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def precision(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2
    return precision


def recall(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    recall = c1 / c3
    return recall


class self_Attention(Layer):
    def __init__(self, **kwargs):
        super(self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Wd = self.add_weight(name='Wd',
                                  shape=(input_shape[2], 1),
                                  initializer='uniform',
                                  trainable=True)
        self.vd = self.add_weight(name='vd',
                                  shape=(input_shape[1], 1),
                                  initializer='uniform',
                                  trainable=True)

        super(self_Attention, self).build(input_shape)

    def call(self, x):
        sentence_output = x

        sd = tf.multiply(tf.tanh(K.dot(sentence_output, self.Wd)), self.vd)  # (batch_size,max_step,max_step)
        sd = tf.squeeze(sd, axis=-1)
        ad = tf.nn.softmax(sd) 
        qd = K.batch_dot(ad, sentence_output)  

        return [qd]

    def compute_output_shape(self, input_shape):
        # assert isinstance(input_shape, list)
        return (input_shape[0], input_shape[2])


embedding_matrix = data_helper.load_pickle('./embedding_matrix.pkl')
embedding_layer = Embedding(embedding_matrix.shape[0],
                            emb_dim,
                            weights=[embedding_matrix],
                            input_length=input_dim,
                            trainable=False)  # mask_zero=True

embedding_matrix_word = data_helper_word.load_pickle('./embedding_matrix_word.pkl')
embedding_layer_word = Embedding(embedding_matrix_word.shape[0],
                                 emb_dim,
                                 weights=[embedding_matrix_word],
                                 input_length=input_dim,
                                 trainable=False)  # mask_zero=True


def encoding_moudle(input_shape, params, deepth, drop):
    sentence = Input(shape=input_shape, dtype='int32')
    sentence_word = Input(shape=input_shape, dtype='int32')

    sentence_embed = embedding_layer(sentence)
    sentence_embed_word = embedding_layer_word(sentence_word)

    encoded_result = []
    sentence_next = sentence_embed
    sentence_next_word = sentence_embed_word
    for i in range(1, deepth + 1):  
        encoder = Bidirectional(LSTM(params["lstm_dim"],
                                     return_sequences=True,
                                     dropout=drop),
                                merge_mode='sum',
                                name="BiLSTM" + str(i))

        sentence_next = encoder(sentence_next)
        sentence_bilstm = Reshape((30, params["lstm_dim"], 1))(sentence_next)
        encoded_result.append(sentence_bilstm)

        sentence_next_word = encoder(sentence_next_word)
        sentence_bilstm_word = Reshape((30, params["lstm_dim"], 1))(sentence_next_word)
        encoded_result.append(sentence_bilstm_word)

    feature_map = encoded_result[0]
    for i in range(len(encoded_result) - 1):
        feature_map = Concatenate(axis=3)([feature_map, encoded_result[i + 1]])

    encode_2DCNN = Conv2D(filters=params["conv2d_dim_encode"],
                          kernel_size=(params["conv2d_kernel_encode_0"], params["conv2d_kernel_encode"]),
                          padding='Valid', strides=[params["conv2d_stride_1_encode"], params["conv2d_stride_2_encode"]],
                          data_format='channels_last',
                          activation='relu')(feature_map)

    encoding_moudle = Model([sentence, sentence_word], encode_2DCNN, name='encoding_moudle')
    encoding_moudle.summary()
    plot_model(encoding_moudle, to_file='encoding_moudle.png', show_shapes=True)
    return encoding_moudle


def siamese_model(params, deepth, drop):
    input_shape = (input_dim,)

    encode_moudel = encoding_moudle(input_shape, params, deepth, drop)

    q1_input = Input(shape=input_shape, dtype='int32', name='sequence1')
    q1_input_word = Input(shape=input_shape, dtype='int32', name='sequence1_word')
    q1_feature = encode_moudel([q1_input, q1_input_word])

    q2_input = Input(shape=input_shape, dtype='int32', name='sequence2')
    q2_input_word = Input(shape=input_shape, dtype='int32', name='sequence2_word')
    q2_feature = encode_moudel([q2_input, q2_input_word])

    attention_dot_match = self_Attention()
    similarity = Concatenate(axis=3)([q1_feature, q2_feature])

    similarity = Conv2D(filters=params["conv2d_dim_match"],
                        kernel_size=(params["conv2d_kernel_match_0"], params["conv2d_kernel_match"]),
                        padding='Valid', strides=[params["conv2d_stride_1_match"], params["conv2d_stride_2_match"]],
                        data_format='channels_last',
                        activation='relu')(similarity)

    similarity = MaxPooling2D(pool_size=(params["pool_kernel_match_0"], params["pool_kernel_match"]),
                              padding="valid", strides=[params["pool_stride_1_match"], params["pool_stride_2_match"]],
                              data_format="channels_last")(similarity)

    _dim1 = backend.int_shape(similarity)[1]
    _dim2 = backend.int_shape(similarity)[2]
    _dim3 = backend.int_shape(similarity)[3]

    similarity = Reshape((_dim1 * _dim2, _dim3,))(similarity)  
    similarity = attention_dot_match(similarity)

    similarity = Dense(1)(similarity)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('sigmoid')(similarity)

    model = Model([q1_input, q1_input_word, q2_input, q2_input_word], [similarity])
    # summarize layers
    model.summary()
    plot_model(model, to_file='framework.png', show_shapes=True)

    op = RMSprop(lr=0.0015)

    from keras.utils import multi_gpu_model
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss="binary_crossentropy", optimizer=op, metrics=['accuracy', precision, recall, f1_score])

    return parallel_model


def train(params, deepth, drop):
    data = data_helper.load_pickle('./model_data.pkl')

    train_q1 = data['train_q1']
    train_q2 = data['train_q2']
    train_y = data['train_label']

    dev_q1 = data['dev_q1']
    dev_q2 = data['dev_q2']
    dev_y = data['dev_label']

    test_q1 = data['test_q1']
    test_q2 = data['test_q2']
    test_y = data['test_label']

    data_word = data_helper_word.load_pickle('./model_data_word.pkl')

    train_q1_word = data_word['train_q1']
    train_q2_word = data_word['train_q2']

    dev_q1_word = data_word['dev_q1']
    dev_q2_word = data_word['dev_q2']

    test_q1_word = data_word['test_q1']
    test_q2_word = data_word['test_q2']

    model = siamese_model(params, deepth, drop)
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True,
                                 mode='max', period=1)
    tensorboard = TensorBoard(log_dir=tensorboard_path)
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=2, mode='max')
    callbackslist = [checkpoint, tensorboard, earlystopping, reduce_lr]

    model.fit([train_q1, train_q1_word, train_q2, train_q2_word], train_y,
              batch_size=512,
              epochs=200,
              verbose=2,
              validation_data=([dev_q1, dev_q1_word, dev_q2, dev_q2_word], dev_y),
              callbacks=callbackslist)

    loss, accuracy, precision, recall, f1_score = model.evaluate([test_q1, test_q1_word, test_q2, test_q2_word], test_y,
                                                                 verbose=1, batch_size=256)


    print("Test best model =loss: %.4f, accuracy:%.4f,precision:%.4f,recall:%.4f,f1_score:%.4f" % (
        loss, accuracy, precision, recall, f1_score))



if __name__ == '__main__':

    params2 = {"lstm_dim": 400,
               "conv2d_dim_encode": 32,
               "conv2d_kernel_encode_0": 3, "conv2d_kernel_encode": 2,
               "conv2d_stride_1_encode": 2, "conv2d_stride_2_encode": 1,
               "conv2d_dim_match": 32,
               "conv2d_kernel_match_0": 2, "conv2d_kernel_match": 2,
               "conv2d_stride_1_match": 1, "conv2d_stride_2_match": 1,
               "pool_kernel_match_0": 3, "pool_kernel_match": 4,
               "pool_stride_1_match": 2, "pool_stride_2_match": 2}

    for drop in range(10):
        print("++++++++++++++++++++++++++++++++0.45"+str(drop))
        train(params2, 2, 0.45)
