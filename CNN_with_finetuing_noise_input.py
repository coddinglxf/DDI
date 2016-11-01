__author__ = 'Administrator'
from Datasets import *
import numpy as np
import keras
import pprint

train_path = "xml/train/"
test_path = "xml/test/"

train_ = Datasets(filename=train_path)
train_data = train_.features
test_ = Datasets(filename=test_path)
test_data = test_.features

word_dict = initial.word_dict
print "get the train data"
train_array = list()
train_label = list()
for feature in train_data:
    if feature['negative'] is False:
        train_label.append(feature['label'])
        train_array.append(feature['all_sequence'])
train_array = np.array(train_array)
print train_array.shape

print "get the train data"
test_array = list()
test_label = list()
for feature in test_data:
    if feature['negative'] is False:
        test_label.append(feature['label'])
        test_array.append(feature['all_sequence'])
test_array = np.array(test_array)
print test_array.shape

from keras.models import Sequential
from keras.layers import *
from keras.layers import Convolution2D, MaxPooling2D, Merge
from keras.optimizers import SGD, adadelta
from keras.utils import np_utils
from keras.constraints import maxnorm
from sklearn.metrics import classification_report
import numpy
from CallBackMy import CallBackMy

for filters in [200, 400]:
    for batchsize in [20, 50]:
        seqList = list()
        for vector_name in word_dict:
            vector = word_dict[vector_name]
            print "vector shape is --->", vector.shape
            seq = Sequential()
            seq.add(Embedding(input_dim=vector.shape[0],
                              output_dim=200,
                              input_length=150,
                              weights=[vector]))
            seq.add(Reshape(dims=(1, 150, vector.shape[1])))
            seq.add(GaussianNoise(0.001))
            # print seq.get_config()
            seqList.append(seq)

        windowsSeqList = list()
        windows = [6, 7, 8, 9]
        for window in windows:
            winSeq = Sequential()
            winSeq.add(Merge(layers=seqList, mode='concat', concat_axis=-3))
            winSeq.add(Reshape(dims=(len(word_dict), 150, 200)))
            winSeq.add(Convolution2D(filters, window, 200, activation='relu'))
            winSeq.add(MaxPooling2D(pool_size=(150 - window + 1, 1)))
            winSeq.add(Flatten())
            windowsSeqList.append(winSeq)

        print "build the model"
        model = Sequential()
        model.add(Merge(windowsSeqList, mode='concat'))
        model.add(Dense(5, W_constraint=maxnorm(100)))
        model.add(Activation('softmax'))

        optimizer = adadelta()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        train_label_cat = np_utils.to_categorical(train_label, 5)
        print train_label_cat.shape
        test_backup = copy.deepcopy(test_label)
        test_label_cat = np_utils.to_categorical(test_label, 5)
        print test_label_cat.shape

        callbackmy = CallBackMy(test_array,
                                windows,
                                test_backup,
                                len(seqList),
                                log_dict={"filters": filters, "batchsize": batchsize},
                                filename=".//log//"+str(filters)+"_"+str(batchsize)+".txt")
        res = model.fit(
            [train_array] * len(seqList),
            train_label_cat,
            validation_data=([test_array] * len(seqList), test_label_cat),
            show_accuracy=True,
            batch_size=batchsize,
            nb_epoch=30,
            callbacks=[callbackmy])

