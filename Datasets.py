# coding=utf-8
__author__ = 'Administrator'
from Document import Sentence, RelationPair
from Initial import Initial
from Tools import load_data
import numpy as np
import math
import copy

print
print "initial only one times----------------------"
initial = Initial(all_data_path="F:\\python_chram_data\\DDI\\all\\")
print "initial only one times----------------------"
print


class Datasets(object):
    def __init__(self, filename="data//test"):
        print "start to load data from path", filename
        self.filename = filename
        self.features = list()
        self.sentences = load_data(filename)
        self.get_features()

    def get_features(self):
        for sentence in self.sentences:
            f = Feature(sen=sentence)
            for instance in f.instances:
                self.features.append(instance)


class Feature(object):
    def __init__(self, sen=Sentence(), padding=1):
        self.sen = sen
        self.words = self.sen.new_context.split("@@")
        self.instances = list()
        self.max_length = 150
        self.features()
        self.padding = padding

    def features(self):
        for relation in self.sen.relation_list:
            instance = dict()
            e1_position = relation.e1_position
            e2_position = relation.e2_position
            padding_words = self.words[e1_position:e2_position + 1]
            padding_words[0] = "DRUG1"
            padding_words[-1] = "DRUG2"
            drug = set()
            for entity in self.sen.entity_list:
                drug.add(entity.text)
            for i in xrange(len(padding_words)):
                if padding_words[i] in drug:
                    padding_words[i] = 'DRUG0'

            # generate all the words
            sentence_words = copy.deepcopy(self.words)
            sentence_words[relation.e1_position] = "DRUG1"
            sentence_words[relation.e2_position] = "DRUG2"
            for i in xrange(len(sentence_words)):
                if sentence_words[i] in drug:
                    sentence_words[i] = "DRUG0"
            # padding the </s>
            left = (self.max_length - len(sentence_words)) / 2
            right = self.max_length - left - len(sentence_words)
            sentence_words_padding = ["</s>"] * left + sentence_words + ["</s>"] * right
            assert len(sentence_words_padding) == self.max_length
            all_sequence = [initial.word2index[word] for word in sentence_words_padding]

            sequence = [initial.word2index[word] for word in padding_words]
            sdp = [initial.word2index[word] for word in relation.sdp.split("@@")]
            # ------->第一个实体和第二个实体之间的单词
            instance['padding_words'] = padding_words
            # ------->整个句子的单词
            instance['word_sequence'] = sentence_words
            # ------->第一个实体和第二个实体之间单词的index
            instance['sequence'] = sequence
            # ------->整个句子单词的index
            instance['all_sequence'] = all_sequence
            # ------->第一个实体在句子中的位置
            instance['e1_pos'] = relation.e1_position
            # ------->第二个实体在句子中的位置
            instance['e2_pos'] = relation.e2_position
            # ------->最短路径，以及对应的index
            instance['sdp'] = sdp
            # ------->关系对应的type，总共有四种
            instance['type'] = relation.type
            # ------->是否存在DDI之间的关系
            instance['ddi'] = relation.ddi
            # ------->type对应的class标号，numpy表示
            instance['class'] = np.array([initial.label[relation.type]])
            # ------->type对应的class标号
            instance['label'] = initial.label[relation.type]
            # ------->对应的二分类时候的标签
            instance['binary'] = 0 if relation.type is "other" else 1
            # ------->对应的二分类时候的标签，numpy表示
            instance['binary_class'] = np.array([instance['binary']])
            # print sequence
            # ------->instance['negative'], 决定是否提前过滤掉该关系
            instance['negative'] = False
            # instance['negative'] = self.filter(instance, relation)
            instance['relation'] = relation
            instance['context'] = self.sen.new_context
            self.instances.append(instance)

    # to decide whether the two entities are illegal
    def filter(self, instance, relation=RelationPair()):
        e1_name = str(relation.e1_name).lower()
        e2_name = str(relation.e2_name).lower()

        return self.filter_1(e1_name, e2_name) \
               or self.filter_2(e1_name, e2_name) \
               or self.filter_3(relation, instance) \
               or self.filter_4(instance=instance)

    # 判断名称是否一样
    def filter_1(self, e1_name, e2_name):
        return e1_name == e2_name

    # 判断一个名称是否是另一个名称的缩写
    def filter_2(self, e1_name, e2_name):
        if len(str(e1_name).split(" ")) > 1:
            if len(str(e2_name).split(" ")) == 1:
                split_words = str(e1_name).split(" ")
                line = "".join([word[0] for word in split_words if str(word).rstrip() != ""])
                return line == e2_name

        if len(str(e2_name).split(" ")) > 1:
            if len(str(e1_name).split(" ")) == 1:
                split_words = str(e2_name).split(" ")
                # print "split words", split_words, "\t", e1_name, e2_name
                line = "".join([word[0] for word in split_words if str(word).rstrip() != ""])
                return line == e1_name

    # 判断 A [and, or, ,, (,] B 的情况
    # 判断 A , or 这种情况
    def filter_3(self, relation=RelationPair(), instance=dict()):
        e1_pos = relation.e1_position
        e2_pos = relation.e2_position
        if math.fabs(e2_pos - e1_pos) == 1:
            return True

        if math.fabs(e2_pos - e1_pos) == 2:
            between = str(self.words[min(e1_pos, e2_pos) + 1]).lower()
            # if between == "and" \
            # or between == "or" \
            #         or between == "," \
            #         or between == "(" \
            #         or between == "-":
            if between == "or" \
                    or between == "," \
                    or between == "(" \
                    or between == "-":
                return True
                # if between == "and" and e1_pos - 1 >= 0:
                #     word = str(instance['word_sequence'][e1_pos - 1]).lower()
                #     if word not in ["of", "between", "with"]:
                #         return True

        if math.fabs(e2_pos - e1_pos) == 3:
            minvalue = min(e1_pos, e2_pos)
            word = str(" ".join(self.words[minvalue + 1: minvalue + 3])).lower()
            if word == ", or" or word == "such as":
                return True

    # filter 掉并列的结构，这个很重要
    # a,b,c, and d
    def filter_4(self, instance=None):
        except_words = [",", 'drug0', 'or', '(', '[', ')', ']', "and"]
        flags = False
        if not instance:
            instance = dict()
        e1_pos = instance['e1_pos']
        e2_pos = instance['e2_pos']
        sequence = instance['word_sequence']
        # print sequence
        for i in xrange(e1_pos + 1, e2_pos):
            word = str(sequence[i]).lower()
            if word not in except_words:
                return False
            else:
                if word == "and":
                    flags = True
        if flags is True:
            if e2_pos - e1_pos <= 4:
                return False
        return True


# dataset = Datasets(filename="e:\\python_chram_data\\DDI\\train\\")
# all_instance = 0
# hash = set()
# sum = dict()
# negative_all = 0
# negative = 0
# negative_wrong = 0
# openfile = open('negative.txt', 'w')
# wrong = open('wrong.txt', 'w')
# writefile = open('svm_train', 'w')
# distance_sum = dict()
# for instance in dataset.features:
#     all_instance += 1
#     # print instance['sdp']
#     # print [initial.index2word[index] for index in instance['sdp']]
#     # print instance['padding_words']
#     # print instance['all_sequence']
#
#     writefile.write(str(instance['label'])
#                     + " "
#                     + instance['type']
#                     + " "
#                     + str(instance['e1_pos'])
#                     + " "
#                     + str(instance['e2_pos'])
#                     + " "
#                     + " ".join(instance['word_sequence']) + "\n")
#
#     hash.add(instance['type'])
#     if instance['type'] in sum:
#         sum[instance['type']] += 1
#     else:
#         sum[instance['type']] = 1
#         # print instance['class']
#     if instance['label'] == 4:
#         negative_all += 1
#         # print instance['negative']
#         if instance['negative'] is False:
#             # print "cool"
#             relation = instance['relation']
#             openfile.write(instance['context'] + "\n")
#             openfile.write(instance['type'] + "\n")
#             openfile.write(relation.e1_name + "\t" + str(relation.e1_position) + "\n")
#             openfile.write(relation.e2_name + "\t" + str(relation.e2_position) + "\n")
#             openfile.write("\n")
#         else:
#             index = instance['e2_pos'] - instance['e1_pos']
#             if index in distance_sum:
#                 distance_sum[index] += 1
#             else:
#                 distance_sum[index] = 1
#
#     if instance['negative'] is True:
#         negative += 1
#         # print "instance['label']---->", instance['label']
#         # print instance['context']
#         # print instance['relation'].smart_print()
#         if instance['class'] != 4:
#             negative_wrong += 1
#             relation = instance['relation']
#             wrong.write(instance['context'] + "\n")
#             wrong.write(instance['type'] + "\n")
#             wrong.write(relation.e1_name + "\t" + str(relation.e1_position) + "\n")
#             wrong.write(relation.e2_name + "\t" + str(relation.e2_position) + "\n")
#             wrong.write("\n")
# openfile.close()
# writefile.close()
# print initial.index2word[51]
# print hash
# print sum
# print all_instance, negative, negative_wrong, negative - negative_wrong, negative_all
# print(distance_sum)
