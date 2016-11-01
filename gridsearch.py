__author__ = 'Administrator'
from Datasets import *
import numpy as np
import keras
import pprint

train_path = "xml\\train\\"
test_path = "xml\\test\\"

train_ = Datasets(filename=train_path)
train_data = train_.features
test_ = Datasets(filename=test_path)
test_data = test_.features

word_dict = initial.word_dict
for each in initial.word_dict:
    vec = initial.word_dict[each]
    word_dict[each] = np.random.uniform(low=-0.1, high=0.1, size=vec.shape)
print "get the train data"
train_array = list()
train_label = list()
for feature in train_data:
    if feature['negative'] is False:
        train_label.append(feature['label'])
        instance_vec = list()
        for filename in word_dict:
            wordvec = word_dict[filename]
            vec = wordvec[feature['all_sequence']]
            instance_vec.append(vec)
        train_array.append(np.array(instance_vec))
train_array = np.array(train_array)
print train_array.shape

print "get the train data"
test_array = list()
test_label = list()
for feature in test_data:
    if feature['negative'] is False:
        test_label.append(feature['label'])
        instance_vec = list()
        for filename in word_dict:
            wordvec = word_dict[filename]
            vec = wordvec[feature['all_sequence']]
            instance_vec.append(vec)
        test_array.append(np.array(instance_vec))
test_array = np.array(test_array)
print test_array.shape

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D, MaxPooling2D, Merge
from keras.optimizers import SGD, adadelta
from keras.utils import np_utils
from keras.constraints import maxnorm
from sklearn.metrics import classification_report
import numpy
from keras.callbacks import EarlyStopping
from CallBackMy import CallBackMy

for max_norm in [100]:
    for filters in [200, 400]:
        windows = [6, 7, 8, 9]
        sequence_list = list()
        for win in windows:
            seq = Sequential()
            seq.add(Convolution2D(filters, win, 200, input_shape=(len(word_dict), 150, 200)))
            seq.add(Activation('tanh'))
            seq.add(MaxPooling2D(pool_size=(150 - win + 1, 1)))
            seq.add(Flatten())
            # seq.add(Dropout(dropout_rate))
            sequence_list.append(seq)

        print "build the model"
        model = Sequential()
        model.add(Merge(sequence_list, mode='concat'))
        model.add(Dense(5, W_constraint=maxnorm(max_norm)))
        model.add(Activation('softmax'))

        optimizer = adadelta()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # print train the model

        # test_label = np_utils.to_categorical(test_label, 5)
        # train = train[0:100]
        # train_label = train_label[0:100]
        #
        # test = test[0:100]
        # test_label = test_label[0:100]
        train_label_cat = np_utils.to_categorical(train_label, 5)
        print train_label_cat.shape

        test_backup = copy.deepcopy(test_label)
        test_label_cat = np_utils.to_categorical(test_label, 5)
        print test_label_cat.shape

        callbackmy = CallBackMy(test_array,
                                windows,
                                test_backup,
                                len(windows),
                                log_dict={"filters": filters, "batchsize": 20},
                                filename=".//log//"+str(filters)+"_"+str(20)+".txt")
        res = model.fit(
            [train_array] * len(windows),
            train_label_cat,
            validation_data=([test_array] * len(windows), test_label_cat),
            show_accuracy=True,
            batch_size=20,
            nb_epoch=20,
            callbacks=[callbackmy])
        # model.evaluate([test, test], test_label)
        preicted = model.predict_classes([test_array] * len(windows))
        print res.history
        print len(preicted)
        print len(test_backup)

        print "max_norm is--->", max_norm, " while filters is --->", filters
        print classification_report(numpy.array(test_backup),
                                    numpy.array(preicted),
                                    digits=4)
