import argparse
import os
import numpy as np
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import scipy.sparse as sp
import pickle
from collections import defaultdict, OrderedDict
import sys
sys.path.append('../')
import src.utils as utils
import time
start = time.time()


POLYSEMIC_WORDS = {'bat', 'virus', 'tank', 'tape', 'bin', 'board',
                   'book', 'bow', 'cap', 'card', 'crane', 'mole',
                   'fan', 'hose', 'keyboard', 'mink', 'mouse',
                   'pipe', 'plug', 'apple'}

class Classify(object):
    def __init__(self, train_embedding_path, eval_embedding_path, train_dir, eval_dir):
        print("Loading train data...")
        self.train_embedding_name = ".".join((os.path.basename(train_embedding_path).strip().split("."))[0:-1])
        print("\t", self.train_embedding_name)
        self.train_E, self.train_i2w, self.train_w2i = self.load_embedding(train_embedding_path)
        self.train_xml2i, self.train_i2xml, self.train_gold_dict_xml, self.train_gold_dict_i = self.load_labels(train_dir, self.train_embedding_name)

        print("Loading eval data...")
        self.eval_embedding_name = ".".join((os.path.basename(eval_embedding_path).strip().split("."))[0:-1])
        print("\t", self.eval_embedding_name)
        self.eval_E, self.eval_i2w, self.eval_w2i = self.load_embedding(eval_embedding_path)
        self.eval_xml2i, self.eval_i2xml, self.eval_gold_dict_xml, self.eval_gold_dict_i = self.load_labels(eval_dir, self.eval_embedding_name)
        print("Extracting training and test data...")
        indices = list(self.eval_gold_dict_i.keys())
        inds = self.eval_w2i['attached']
        xmls = [self.eval_gold_dict_i[i] for i in inds if i in self.eval_gold_dict_i.keys()]
        print("wordforms: ", xmls)
        words = [self.eval_i2w[i] for i in indices]
        print("Number of words: ", len(words))
        words = set(words)
        print("Number of dfferent words: ", len(words))
        all_test_truth = []
        all_pred = defaultdict(list)
        for w in words:
            X_train, y_train, X_test, y_test, le, eval_prefix_xmls = self.get_data(word=w)
            assert X_test.shape[0] != 0
            if len(set(y_train)) == 1 and len(y_test) > 0:
                print("###########################\n", w, "(1 train)\n###########################")
                all_test_truth.extend(self.format_labels(le.inverse_transform(y_test), eval_prefix_xmls))
                y_pred = [le.inverse_transform([y_train[0]])[0] for i in range(len(y_test))]
                # all_pred['AdaBoostClassifier'].extend(self.format_labels(y_pred, eval_prefix_xmls))
                # all_pred['KNeighborsClassifier'].extend(self.format_labels(y_pred, eval_prefix_xmls))
                all_pred['LogReg'].extend(self.format_labels(y_pred, eval_prefix_xmls))
            elif X_train.shape[0] == 0:
                print("###########################\n", w, "(0 train)\n###########################")
                all_test_truth.extend(self.format_labels(le.inverse_transform(y_test), eval_prefix_xmls))
                y_pred = ["NONE" for i in range(len(y_test))]
                # all_pred['AdaBoostClassifier'].extend(self.format_labels(y_pred, eval_prefix_xmls))
                # all_pred['KNeighborsClassifier'].extend(self.format_labels(y_pred, eval_prefix_xmls))
                all_pred['LogReg'].extend(self.format_labels(y_pred, eval_prefix_xmls))
            elif X_test.shape[0] == 0:
                print("0 test shape!!!!!!", w)
            else: #elif X_train.shape[0] > 0 and X_test.shape[0] > 0: # and len(le.classes_) > 1 and len(set(y_train)) > 1:
                print("###########################\n", w, "\n###########################")
                all_test_truth.extend(self.format_labels(le.inverse_transform(y_test), eval_prefix_xmls))
                # print("Classes: ", le.classes_)
                self.do_all_classifiers(X_train, y_train, X_test, y_test, all_pred, le, eval_prefix_xmls)
        # print("SVM: ", f1_score(all_test_truth, all_pred['SVM'], average='micro'))
        # print("GaussianProcessClassifier: ",  f1_score(all_test_truth, all_pred['GaussianProcessClassifier'], average='micro'))
        # print("DecisionTreeClassifier: ",  f1_score(all_test_truth, all_pred['DecisionTreeClassifier'], average='micro'))
        # print("RandomForestClassifier: ",  f1_score(all_test_truth, all_pred['RandomForestClassifier'], average='micro'))
        # print("MLPClassifier: ",  f1_score(all_test_truth, all_pred['MLPClassifier'], average='micro'))
        # print("AdaBoostClassifier: ",  f1_score(all_test_truth, all_pred['AdaBoostClassifier'], average='micro'))
        # print("KNeighborsClassifier: ",  f1_score(all_test_truth, all_pred['KNeighborsClassifier'], average='micro'))
        print("LogReg: ",  f1_score(all_test_truth, all_pred['LogReg'], average='micro'))
        self.save_pred(all_test_truth, all_pred)
        print("Number of eval: ", len(all_test_truth))


    def format_labels(self, inverse_labels, prefix_xmls):
        assert len(inverse_labels) == len(prefix_xmls)
        formatted_labels = []
        for i in range(len(inverse_labels)):
            text = prefix_xmls[i] + " " + inverse_labels[i]
            formatted_labels.append(text)
        # print(formatted_labels)
        return formatted_labels

    def save_pred(self, all_truth, all_pred):
        all_test_path = "../results/classifier/" + self.eval_embedding_name + "/gold_all.txt"
        if not os.path.exists(os.path.dirname(all_test_path)):
            os.makedirs(os.path.dirname(all_test_path))
        truth_file = open(all_test_path, "w")
        for e in all_truth:
            truth_file.write(e+'\n')

        for clf, pred in all_pred.items():
            print("Saving ", clf, " classifier predictions...")
            all_pred_clf_path = "../results/classifier/" + self.eval_embedding_name + "/" + clf + "_pred.txt"
            if not os.path.exists(os.path.dirname(all_pred_clf_path)):
                os.makedirs(os.path.dirname(all_pred_clf_path))
            pred_clf_file = open(all_pred_clf_path, "w")
            for e in pred:
                pred_clf_file.write(e + '\n')

    def load_labels(self, dir, name):
        xml2i = pickle.load(open(dir+name+"_xml2i.p", "rb"))
        i2xml = pickle.load(open(dir+name+"_i2xml.p", "rb"))
        gold_dict_xml = pickle.load(open(dir+name+"_gold_dict_xml.p", "rb"))
        gold_dict_i = pickle.load(open(dir+name+"_gold_dict_i.p", "rb"))
        return xml2i, i2xml, gold_dict_xml, gold_dict_i

    def load_embedding(self, embedding_path):
        name = ".".join((os.path.basename(embedding_path).strip().split("."))[0:-1])
        i2w = pickle.load(open('../data/indexing/words/embeddings/' + name + "_i2w.p", 'rb'))
        # w2i = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_w2i.p", 'rb'))
        w2i = defaultdict(set)
        for i, w in i2w.items():
            w2i[w].add(i)
        E = sp.load_npz(embedding_path)
        E = E.tocsr()
        # ordered = sorted(i2w.items(), key=lambda t: int(t[0]))
        # print(ordered[0:20])
        return E, i2w, w2i

    def get_relevant_ids(self, word, train=True, dict=None):
        """
        Get relevant xml IDs ie IDs that contain the argument "word"
        :param word: word to search for
        :param dict: gold dict conainting (xml_id, labels) key-value pairs
        :return: set of relevant xml IDs
        """
        relevant_xmlIDs = set()
        relevant_xmlIDs_new = set()
        if train:
            train_keys = set(self.train_gold_dict_i.keys())
            inds = self.train_w2i[word]
            for i in inds:
                if i in train_keys:
                    relevant_xmlIDs_new.add(self.train_i2xml[i])
        else: # eval
            eval_keys = set(self.eval_gold_dict_i.keys())
            inds = self.eval_w2i[word]
            for i in inds:
                if i in eval_keys:
                    relevant_xmlIDs_new.add(self.eval_i2xml[i])
        # for xmlID, labels in dict.items():
        #     for label in labels:
        #         if label.startswith(word + '%'):
        #             relevant_xmlIDs.add(xmlID)
        #             break
        # print(relevant_xmlIDs, "\n", relevant_xmlIDs_new)
        return relevant_xmlIDs_new

    def get_data(self, word="watch"):
        # get only the first sense
        # outdated: some indices may appear twice due to ambiguous gold data!!!

        train_relevant_xmlIDs = self.get_relevant_ids(word, True, self.train_gold_dict_xml)
        eval_relevant_xmlIDs = self.get_relevant_ids(word, False, self.eval_gold_dict_xml)
        train_relevant_ids = [self.train_xml2i[xml] for xml in train_relevant_xmlIDs if xml in set(self.train_xml2i.keys())]
        eval_relevant_ids = [self.eval_xml2i[xml] for xml in eval_relevant_xmlIDs if xml in set(self.eval_xml2i.keys())]
        eval_relevant_xmls = [self.eval_i2xml[i] for i in eval_relevant_ids]

        train_indices = sorted(train_relevant_ids)
        eval_indices = sorted(eval_relevant_ids)
        train_X = self.train_E.tocsr()[train_indices, :].todense()
        eval_X = self.eval_E.tocsr()[eval_indices, :].todense()
        raw_train_y = [' '.join(self.train_gold_dict_xml[self.train_i2xml[i]]) for i in train_relevant_ids]
        # current: takes all senses
        raw_eval_y = [' '.join(self.eval_gold_dict_xml[self.eval_i2xml[i]]) for i in eval_relevant_ids]
        # truth_format_eval_y = [self.eval_i2xml[i] + " " + self.eval_gold_dict_xml[self.eval_i2xml[i]][0] for i in eval_relevant_ids]

        le = pp.LabelEncoder()
        le.fit(np.concatenate((raw_train_y, raw_eval_y)))
        train_y = le.transform(raw_train_y)
        eval_y = le.transform(raw_eval_y)
        # print("\ttrain: ", train_X.shape, train_y.shape)
        # print("\teval: ", eval_X.shape, eval_y.shape)
        return train_X, train_y, eval_X, eval_y, le, eval_relevant_xmls

    def classify(self, classifier, X_train, y_train, X_test, y_test):
        clf = classifier.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # print("\tpredicted:\t", y_pred, "\n\ttrue:\t", y_test)
        # print(classification_report(y_test, y_pred))
        # fscore = f1_score(y_test, y_pred, average='micro')
        # print("micro fscore: ", fscore)
        # score = clf.score(X_test, y_test)
        # print("score: ", score)
        return y_pred

    def do_all_classifiers(self, X_train, y_train, X_test, y_test, all_pred, le, eval_prefix_xmls):
        # print("----------------------------------\nSVC")
        # y_pred = self.classify(SVC(kernel="linear", C=0.025, class_weight="balanced"), X_train, y_train, X_test, y_test)
        # all_pred['SVM'].extend(self.format_labels(le.inverse_transform(y_pred), eval_prefix_xmls))
        # print("----------------------------------\nGaussianProcessClassifier")
        # y_pred = self.classify(GaussianProcessClassifier(1.0 * RBF(1.0)), X_train, y_train, X_test, y_test)
        # all_pred['GaussianProcessClassifier'].extend(self.format_labels(le.inverse_transform(y_pred), eval_prefix_xmls))
        # print("----------------------------------\nDecisionTreeClassifier")
        # y_pred = self.classify(DecisionTreeClassifier(max_depth=5, class_weight="balanced"), X_train, y_train, X_test, y_test)
        # all_pred['DecisionTreeClassifier'].extend(self.format_labels(le.inverse_transform(y_pred), eval_prefix_xmls))
        # print("----------------------------------\nRandomForestClassifier")
        # y_pred = self.classify(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, class_weight="balanced"), X_train, y_train, X_test, y_test)
        # all_pred['RandomForestClassifier'].extend(self.format_labels(le.inverse_transform(y_pred), eval_prefix_xmls))
        # print("----------------------------------\nMLPClassifier")
        # y_pred = self.classify(MLPClassifier(alpha=1, max_iter=1000), X_train, y_train, X_test, y_test)
        # all_pred['MLPClassifier'].extend(le.inverse_transform(y_pred))
        # print("----------------------------------\nAdaBoostClassifier")
        # y_pred = self.classify(AdaBoostClassifier(random_state=1), X_train, y_train, X_test, y_test)
        # all_pred['AdaBoostClassifier'].extend(self.format_labels(le.inverse_transform(y_pred), eval_prefix_xmls))
        # print("----------------------------------\nKNeighborsClassifier")
        # if len(y_train) < 3 or len(y_test) < 3:
        #     y_pred = self.classify(KNeighborsClassifier(1, n_jobs=3), X_train, y_train, X_test, y_test)
        # else:
        #     y_pred = self.classify(KNeighborsClassifier(3, n_jobs=3), X_train, y_train, X_test, y_test)
        # all_pred['KNeighborsClassifier'].extend(self.format_labels(le.inverse_transform(y_pred), eval_prefix_xmls))
        # print("----------------------------------\nLogReg")
        y_pred = self.classify(LogisticRegression(class_weight='balanced', random_state=1, solver='liblinear'), X_train, y_train, X_test, y_test)
        all_pred['LogReg'].extend(self.format_labels(le.inverse_transform(y_pred), eval_prefix_xmls))
        # print("----------------------------------\nGaussianNB")
        # y_pred = self.classify(GaussianNB(), X_train, y_train, X_test, y_test)
        # all_pred['GaussianNB'].extend(self.format_labels(le.inverse_transform(y_pred), eval_prefix_xmls))

    def get_recurring_words(self):
        """
        Get words that have multiple embeddings.
        :return:  Dictionary of words with multiple embeddings in a form: (word: {set of indices})
        """
        recurring = defaultdict(set)
        for word, inds in self.train_w2i.items():
            if len(inds) > 1:
                recurring[word] = inds
        print('Out of', len(self.train_w2i.keys()), 'words there is', len(recurring.keys()), 'with multiple forms.')
        return recurring


def main():
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--trainEmbedding', required=False,
                            default='../data/sparse_matrices/word_base/embeddings/2000_0.5_train_alphas.txt.gz.npz')
        parser.add_argument('--evalEmbedding', required=False,
                            default='../data/sparse_matrices/word_base/embeddings/2000_0.5_eval_alphas.txt.gz.npz')
        parser.add_argument('--trainDir',required=False,
                            default='../data/indexing/gold_labels/train/')
        parser.add_argument('--evalDir',required=False,
                            default='../data/indexing/gold_labels/eval/all/')
        args = parser.parse_args()
        print("The command line arguments were ", args)
        train_name = args.trainEmbedding
        eval_name = train_name.replace("train", "ALL")
        print("TRAIN:", train_name, "\nEVAL:", eval_name)
        Classify(train_name, eval_name, args.trainDir, args.evalDir)
        end = time.time()
        runtime = end-start

        print("Runtime: ", str(runtime), "seconds (", str(runtime/3600), " hours)")

if __name__ == "__main__":
    main()