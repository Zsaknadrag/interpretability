import gzip
import argparse
import os
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics.cluster import v_measure_score
import scipy.sparse as sp
import pickle
from collections import defaultdict, OrderedDict
import sys
sys.path.append('../')
import src.utils as utils

POLYSEMIC_WORDS = {'watch','bat', 'virus', 'tank', 'tape', 'bin', 'board',
                   'book', 'bow', 'cap', 'card', 'crane', 'mole',
                   'fan', 'hose', 'keyboard', 'mink', 'mouse',
                   'pipe', 'plug', 'apple', 'watch'}

class Cluster(object):
    def __init__(self, embedding_path, dir_path):
        self.embedding_name = ".".join((os.path.basename(embedding_path).strip().split("."))[0:-1])
        self.E, self.i2w, self.w2i = self.load_embedding(embedding_path)
        self.xml2i, self.i2xml, self.gold_dict_xml, self.gold_dict_i = self.load_labels(dir_path, self.embedding_name)
        self.cluster()

    def load_embedding(self, embedding_path):
        i2w = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_i2w.p", 'rb'))
        # w2i = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_w2i.p", 'rb'))
        w2i = defaultdict(set)
        for i, w in i2w.items():
            w2i[w].add(i)
        E = sp.load_npz(embedding_path)
        E = E.tocsr()
        return E, i2w, w2i

    def load_labels(self, dir, name):
        xml2i = pickle.load(open(dir+name+"_xml2i.p", "rb"))
        i2xml = pickle.load(open(dir+name+"_i2xml.p", "rb"))
        gold_dict_xml = pickle.load(open(dir+name+"_gold_dict_xml.p", "rb"))
        gold_dict_i = pickle.load(open(dir+name+"_gold_dict_i.p", "rb"))
        return xml2i, i2xml, gold_dict_xml, gold_dict_i

    def get_recurring_words(self):
        """
        Get words that have multiple embeddings.
        :return:  Dictionary of words with multiple embeddings in a form: (word: {set of indices})
        """
        recurring = defaultdict(set)
        for word, inds in self.w2i.items():
            if len(inds) > 1:
                recurring[word] = inds
        print('Out of', len(self.w2i.keys()), 'words there is', len(recurring.keys()), 'with multiple forms.')
        return recurring

    def cluster(self):
        recurring = self.get_recurring_words()
        avg_vmeasure = defaultdict(float)
        for word, inds in recurring.items():
            gold_inds = []
            gold_keys = set(self.gold_dict_i.keys())
            for i in inds:
                if i in gold_keys:
                    gold_inds.append(i)
            indices = sorted(gold_inds)
            if len(indices) > 2 and len(indices) < 3000:# and word in POLYSEMIC_WORDS:
                print(word, len(indices))
                D = self.E.tocsr()[indices, :]
                # self.SpectralClustering_(D, indices)

                true_labels = self.get_true_labels(indices)
                number_of_clusters = len(set(true_labels))
                # print("number of clusters: ", number_of_clusters)
                labels_kmeans = self.KMeans_(D, indices, number_of_clusters)
                labels_aggl = self.AgglomerativeClustering_(D, indices, number_of_clusters)
                labels_spectral = self.SpectralClustering_(D, indices, number_of_clusters)

                KMeans_vmeasure = v_measure_score(true_labels, labels_kmeans)
                aggl_vmeasure = v_measure_score(true_labels, labels_aggl)
                spectral_vmeasure = v_measure_score(true_labels, labels_spectral)
                # print("Kmeans vmeasure: ", KMeans_vmeasure)
                avg_vmeasure["Kmeans"] += KMeans_vmeasure
                avg_vmeasure["Aggl"] += aggl_vmeasure
                avg_vmeasure["Spectral"] += spectral_vmeasure
                avg_vmeasure["Denom"] += 1
        print("Average kmeans vmeasure: ", avg_vmeasure["Kmeans"]/avg_vmeasure["Denom"])
        print("Average Aggl vmeasure: ", avg_vmeasure["Aggl"] / avg_vmeasure["Denom"])
        print("Average Spectral vmeasure: ", avg_vmeasure["Spectral"] / avg_vmeasure["Denom"])

    def get_true_labels(self, indices):
        true_labels = []
        for i in indices:
            true_labels.append(self.gold_dict_i[i][0])
        return true_labels

    def get_context(self, index, context_size):
        # get context
        text = []
        # print('\t', end="")
        for i in range(index - context_size, index + context_size + 1):
            if i >= 0 and i < len(self.i2w.keys()):
                # print(self.i2w[i], end=" ")
                text.append(self.i2w[i])
        # print('\n')
        return " ".join(text)

    def save_clustering(self, name, indices, labels, context_size=10):
        dir_name = "../results/clustering/" + self.embedding_name + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file = open(os.path.join(dir_name+name), "a")
        for i in range(len(indices)):
            # print(self.i2w[indices[i]], labels[i])
            text = self.get_context(indices[i], context_size)
            file.write(str(i)+" "+self.i2w[indices[i]]+" label_"+str(labels[i])+"\n\t"+text+"\n")
        file.write("\n")

    def KMeans_(self, X, indices, n_clusters):
        clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        labels = clustering.labels_
        context_size = 10
        self.save_clustering("Kmeans.txt", indices, labels, context_size)
        # for i in range(len(indices)):
        #     print(self.i2w[indices[i]], labels[i])
        #     self.get_context(indices[i], context_size)
        return labels

    def SpectralClustering_(self, X, indices, n_clusters):
        clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0).fit(X)
        labels = clustering.labels_
        context_size = 10
        self.save_clustering("SpectralClustering.txt", indices, labels, context_size)
        # for i in range(len(indices)):
        #     print(self.i2w[indices[i]], labels[i])
        #     self.get_context(indices[i], context_size)
        return labels

    def AgglomerativeClustering_(self, X, indices, n_clusters):
        clustering = AgglomerativeClustering(affinity="cosine", linkage="average",
                                             compute_full_tree=True, n_clusters=n_clusters).fit(X.todense())
        labels = clustering.labels_
        context_size = 10
        self.save_clustering("AgglomerativeClustering.txt", indices, labels, context_size)
        # for i in range(len(indices)):
        #     print(self.i2w[indices[i]], labels[i])
        #     self.get_context(indices[i], context_size)
        return labels

def main():
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--embedding', required=False,
                            default='../data/sparse_matrices/word_base/embeddings/2000_0.1_train_alphas.txt.gz.npz', type=str)
        parser.add_argument('--dir', required=False,
                            default='../data/indexing/gold_labels/train/')
        args = parser.parse_args()
        print("The command line arguments were ", args)

        Cluster(args.embedding, args.dir)
if __name__ == "__main__":
    main()