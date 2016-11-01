__author__ = 'aci'
from keras.callbacks import Callback
from sklearn.metrics import classification_report, f1_score
from MicroCalculate import *


class CallBackMy(Callback):
    def __init__(self, test_array, windows, test_backup, input_length, log_dict={},
                 caseStudy=False,
                 test_features=list(),
                 filename="",
                 function=None):
        super(CallBackMy, self).__init__()
        self.log_dict = log_dict
        self.test_array = test_array
        self.windows = windows
        self.test_backup = test_backup
        self.input_length = input_length
        self.caseStudy = caseStudy
        self.test_features = test_features
        self.filename = filename

        # output = function([self.test_array]*len(self.input_length))
        # print output

    def on_epoch_end(self, batch, logs={}):
        print "this is the end of the data, and test data on every epoch", self.log_dict
        predicted = self.model.predict_classes([self.test_array] * self.input_length)
        p, r, f = calculateMicroValue(y_pred=predicted, y_true=self.test_backup, labels=[0, 1, 2, 3])
        openfile = open(self.filename, 'a')
        openfile.write(str(p) + "\t" + str(r) + "\t" + str(f) + "\r\n")
        # calculateMicroValue(y_pred=predicted, y_true=self.test_backup, labels=[0, 1, 2, 3, 4])
        rep = classification_report(
            y_pred=predicted,
            y_true=self.test_backup,
            digits=4,
        )
        openfile.write(rep + "\n")
        openfile.close()
        print rep
        if self.caseStudy:
            predicted = self.model.predict_classes([self.test_array] * self.input_length)
            openfile = open("caseStudy//"+str(self.log_dict['filters']) + "_" + str(self.log_dict['batchsize']), 'w')
            openfile.close()
            openfile = open("caseStudy//"+str(self.log_dict['filters']) + "_" + str(self.log_dict['batchsize']), 'a')
            for i in xrange(len(self.test_backup)):
                predicted_label = predicted[i]
                pretrained_label = self.test_backup[i]
                if predicted_label != pretrained_label:
                    openfile.write(str(predicted_label) + "\t" + str(pretrained_label) + "\n")
                    openfile.write(str(self.test_features[i]["word_sequence"]) + "\n")
                    openfile.write("the true label for this instace is--->" + str(self.test_features[i]['label']) + "\n")
                    openfile.write("\n")
            openfile.close()