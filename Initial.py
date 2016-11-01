__author__ = 'Administrator'
from Tools import load_data
import numpy as np


class Initial(object):
    def __init__(self,
                 all_data_path,
                 vector_length=200,
                 pre_trained_embedding=".//data//wiki_pubmed"):
        self.all_data_path = all_data_path
        self.pre_trained_embedding = pre_trained_embedding

        # word2index and index2word
        self.word2index = dict()
        self.index2word = dict()
        self.word2index['</s>'] = 0
        self.word2index['DRUG1'] = 1
        self.word2index['DRUG2'] = 2
        self.word2index['DRUG0'] = 3
        self.index()

        # init the word vectors
        self.word_dict = dict()
        self.vector_length = vector_length
        filename_list = ['.//data//pubmed',
                         './/data//pubmed_and_pmc',
                         './/data//wiki_pubmed',
                         './/data//pmc',
                         './/data//pubmed_myself',
                         ]
        # filename_list = ['.//data//wiki_pubmed']
        for filename in filename_list:
            self.word_dict[filename] = self.init_word_embedding(filename)

        # init the label
        self.label = dict()
        self.label['int'] = 0
        self.label['advise'] = 1
        self.label['effect'] = 2
        self.label['mechanism'] = 3
        self.label['other'] = 4

    def index(self):
        print "start index the data in ", self.all_data_path
        sentences = load_data(path=self.all_data_path)
        current_index = 4
        for sentence in sentences:
            words = str(sentence.new_context).strip("\r").strip("\n").rstrip().split("@@")
            for word in words:
                if word not in self.word2index:
                    self.word2index[word] = current_index
                    current_index += 1
            # index the relations
            for relation in sentence.relation_list:
                sdps = str(relation.sdp).split("@@")
                for sdp in sdps:
                    if sdp not in self.word2index:
                        self.word2index[sdp] = current_index
                        current_index += 1

        for word in self.word2index:
            self.index2word[self.word2index[word]] = word

        # assert to make sure
        for current_index in self.index2word:
            assert self.word2index[self.index2word[current_index]] == current_index

    def init_word_embedding(self, filename):
        # init the pre_trained_embedding from text
        print "start to load the pre trained word embedding from", filename
        Word = np.random.uniform(low=-0.1, high=0.1, size=(len(self.word2index), self.vector_length))
        openfile = open(filename)
        word_2_vector = dict()
        for line in openfile:
            words = str(line).strip("\r\n").split(" ")
            word_2_vector[words[0]] = [float(words[i]) for i in xrange(1, len(words))]
        openfile.close()

        word_2_vector['</s>'] = [0.0] * self.vector_length
        # init the self.Words
        word_in_pretrained = 0
        for i in xrange(len(self.word2index)):
            word = self.index2word[i]
            if word in word_2_vector:
                Word[i] = word_2_vector[word]
                word_in_pretrained += 1
        print "all the word size is --->", len(self.word2index)
        print "words in pre-trained word embedding is ---# >", word_in_pretrained
        return Word

