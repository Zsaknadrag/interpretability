import gzip
import argparse
import os
import numpy as np
import xml.etree.ElementTree as ET
from sklearn import preprocessing as pp

import scipy.sparse as sp
import pickle
from collections import defaultdict, OrderedDict
import sys
sys.path.append('../')
import src.steps.utils as utils
import time

POLYSEMIC_WORDS = {'bat', 'virus', 'tank', 'tape', 'bin', 'board',
                   'book', 'bow', 'cap', 'card', 'crane', 'mole',
                   'fan', 'hose', 'keyboard', 'mink', 'mouse',
                   'pipe', 'plug', 'apple'}

class TrainXmlProcessor(object):
    def __init__(self, train_embedding_path, train_dir_path):
        self.embedding_name = ".".join((os.path.basename(train_embedding_path).strip().split("."))[0:-1])
        print("Loading data...")
        self.E, self.i2w, self.w2i = self.load_embedding(train_embedding_path)
        print("Saving data...")
        self.xmlID2i, self.i2xmlID, self.gold_dict = self.load_labels(train_dir_path)

    def load_labels(self, dir_path):
        for file_name in os.listdir(dir_path):
            # xml
            if file_name.find('.xml') != -1:
                xml_file = open(os.path.join(dir_path, file_name), 'r')
                xmlID2i = self.process_xml(xml_file)
            # labels
            if file_name.find('.txt') != -1:
                gold_file = open(os.path.join(dir_path, file_name), 'r')
                gold_dict = self.get_gold_dict(gold_file)
        i2xmlID = {xml: i for i, xml in xmlID2i.items()}
        gold_dict_i = {xmlID2i[xml]: label for xml, label in gold_dict.items() if xml in set(xmlID2i.keys())}
        assert gold_dict!= None
        assert xmlID2i != None
        out_dir = "../data/indexing/gold_labels/train/"
        utils.pickler(out_dir+self.embedding_name+"_xml2i.p", xmlID2i)
        utils.pickler(out_dir+self.embedding_name+"_i2xml.p", i2xmlID)
        utils.pickler(out_dir+self.embedding_name+"_gold_dict_xml.p", gold_dict)
        utils.pickler(out_dir + self.embedding_name + "_gold_dict_i.p", gold_dict_i)
        return xmlID2i, i2xmlID, gold_dict

    def process_xml(self, xml_file):
        xml = xml_file.readlines()
        sentence = ''
        id_counter = 0
        xmlID2i = {}
        for i in range(len(xml)):
            line = xml[i]
            if line.find('<sentence id=') != -1:
                sentence = ''
                sentence += line
            elif line.find('</sentence>') != -1 and id_counter < len(self.i2w):
                sentence += line
                # process xml
                root = ET.fromstring(sentence)
                id_counter = self.process_xml_root(root, xmlID2i, id_counter)
                sentence = ''
            else:
                sentence += line
        return xmlID2i

    def process_xml_root(self, root, xmlID2i, id_counter):
        for child in root:
            if child.tag == 'wf':
                word = child.text
                # print(id_counter, word, self.i2w[id_counter])
                assert word == self.i2w[id_counter]
            if child.tag == 'instance':
                word = child.text
                word = word.replace(' ', '_')
                assert word == self.i2w[id_counter]
                id = child.get('id')
                assert id not in xmlID2i.keys()
                xmlID2i[id] = id_counter
                # print(word, self.i2w[id_counter], id)
            id_counter += 1
        return id_counter

    def get_gold_dict(self, gold_file):
        gold_dict = defaultdict(list)
        for line in gold_file:
            parts = line.strip().split(' ')
            assert len(parts) > 1
            ind = parts[0]
            labels = list(parts[1:])
            gold_dict[ind] = labels
        return gold_dict

    def load_embedding(self, embedding_path):
        i2w = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_i2w.p", 'rb'))
        # w2i = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_w2i.p", 'rb'))
        w2i = defaultdict(set)
        for i, w in i2w.items():
            w2i[w].add(i)
        E = sp.load_npz(embedding_path)
        E = E.tocsr()
        # ordered = sorted(i2w.items(), key=lambda t: int(t[0]))
        return E, i2w, w2i

class EvalXmlProcessor(object):
    def __init__(self, eval_embedding_path, eval_dir):
        self.embedding_name = ".".join((os.path.basename(eval_embedding_path).strip().split("."))[0:-1])
        print("Loading data...")
        self.E, self.i2w, self.w2i = self.load_embedding(eval_embedding_path)
        print("Saving data...")
        self.load_labels(eval_dir)
        # self.xmlID2i, self.i2xmlID, self.gold_dict = self.load_labels(train_dir_path)

    def load_embedding(self, embedding_path):
        i2w = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_i2w.p", 'rb'))
        w2i = defaultdict(set)
        for i, w in i2w.items():
            w2i[w].add(i)
        E = sp.load_npz(embedding_path)
        E = E.tocsr()
        print("Embedding shape: ", E.shape)
        return E, i2w, w2i

    def load_labels(self, eval_dir):
        # order: seneval2, senseval3, semeval2007, semeval2013, semeval2015
        all_xml2i, all_i2xml, all_gold_dict, all_gold_dict_i = {}, {}, {}, {}
        sub_dirs = []
        for sub_dir in os.listdir(eval_dir):
            sub_dirs.append(eval_dir+sub_dir)
        id_counter = 0
        for sub_dir in sub_dirs:
            eval_name = os.path.basename(sub_dir)
            print(eval_name)
            for file_name in os.listdir(sub_dir):
                # xml
                if file_name.find('.xml') != -1:
                    xml_file = open(os.path.join(sub_dir, file_name), 'r')
                    xml2i, id_counter = self.process_xml(xml_file, eval_name, id_counter)
                # labels
                if file_name.find('.txt') != -1:
                    gold_file = open(os.path.join(sub_dir, file_name), 'r')
                    gold_dict = self.get_gold_dict(gold_file, eval_name)
            i2xml = {xml: i for i, xml in xml2i.items()}
            gold_dict_i = {xml2i[xml]: label for xml, label in gold_dict.items() if xml in set(xml2i.keys())}
            assert gold_dict != None
            assert xml2i != None
            out_dir = "../data/indexing/gold_labels/eval/" + eval_name + "/"
            self.save_data(out_dir, xml2i, i2xml, gold_dict, gold_dict_i)
            self.gather_data(xml2i, all_xml2i)
            self.gather_data(i2xml, all_i2xml)
            self.gather_data(gold_dict, all_gold_dict)
            self.gather_data(gold_dict_i, all_gold_dict_i)
        print("ALL")
        all_out_dir = "../data/indexing/gold_labels/eval/all/"
        self.save_data(all_out_dir, all_xml2i, all_i2xml, all_gold_dict, all_gold_dict_i)

    def gather_data(self, data, dest):
        for k, v in data.items():
            dest[k] = v

    def save_data(self, out_dir, xml2i, i2xml, gold_dict, gold_dict_i):
        utils.pickler(out_dir + self.embedding_name + "_xml2i.p", xml2i)
        utils.pickler(out_dir + self.embedding_name + "_i2xml.p", i2xml)
        utils.pickler(out_dir + self.embedding_name + "_gold_dict_xml.p", gold_dict)
        utils.pickler(out_dir + self.embedding_name + "_gold_dict_i.p", gold_dict_i)

    def process_xml(self, xml_file, prefix, id_counter):
        xml = xml_file.readlines()
        sentence = ''
        # id_counter = 0
        xml2i = {}
        for i in range(len(xml)):
            line = xml[i]
            if line.find('<sentence id=') != -1:
                sentence = ''
                sentence += line
            elif line.find('</sentence>') != -1 and id_counter < len(self.i2w):
                sentence += line
                # process xml
                root = ET.fromstring(sentence)
                id_counter = self.process_xml_root(root, xml2i, prefix, id_counter)
                sentence = ''
            else:
                sentence += line
        return xml2i, id_counter

    def process_xml_root(self, root, xml2i, prefix, id_counter):
        for child in root:
            if child.tag == 'wf':
                word = child.text
                assert word == self.i2w[id_counter]
            if child.tag == 'instance':
                word = child.text
                word = word.replace(' ', '_')
                assert word == self.i2w[id_counter]
                id = child.get('id')
                id = prefix+"_"+id
                assert id not in xml2i.keys()
                xml2i[id] = id_counter
            id_counter += 1
        return id_counter


    def get_gold_dict(self, file, prefix):
        gold_dict = defaultdict(list)
        for line in file:
            parts = line.strip().split(' ')
            assert len(parts) > 1
            ind = prefix+"_"+parts[0]
            labels = list(parts[1:])
            gold_dict[ind] = labels
        return gold_dict

def main():
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--trainEmbedding', required=True,
                            default='../data/sparse_matrices/word_base/embeddings/2000_0.5_train_alphas.txt.gz.npz',
                            help='Path to npz format contextual sparse embedding used for training.')
        parser.add_argument('--trainDir', required=True,
                            default='../data/wsd/train/SemCor', help='Path to directory containing wsd training files.')
        parser.add_argument('--evalEmbedding', required=False,
                            default='../data/sparse_matrices/word_base/embeddings/2000_0.5_eval_alphas.txt.gz.npz',
                            help='Path to npz format contextual sparse embedding used for evaluation.')
        parser.add_argument('--evalDir',required=False,
                            default='../data/wsd/eval/', help='Path to directory containing wsd eval files.')

        args = parser.parse_args()
        print("The command line arguments were ", args)
        start = time.time()
        TrainXmlProcessor(args.trainEmbedding, args.trainDir)
        EvalXmlProcessor(args.evalEmbedding, args.evalDir)
        end = time.time()
        runtime = end-start
        print("Runtime: ", str(runtime), "seconds (", str(runtime/3600), " hours)")


if __name__ == "__main__":
    main()