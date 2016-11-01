__author__ = 'Administrator'
from Document import Document
from progressbar import *
import time
import os


# load data from a path
def load_data(path=""):
    time.sleep(0.5)
    print "start to load data from path----->", path
    time.sleep(0.5)
    file_list = os.listdir(path)
    sentences = list()
    widgets = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
    pbar = ProgressBar(widgets=widgets, maxval=len(file_list)).start()
    for i in xrange(len(file_list)):
        filename = file_list[i]
        current_path = os.path.join(path, filename)
        document = Document(filename=current_path)
        for sentence in document.sentence_list:
            sentences.append(sentence)
        pbar.update(i+1)
    time.sleep(1)
    print
    return sentences

def isNumber(inputString):
    pass


# load_data(path="data//drug_and_medline_tidy//")
