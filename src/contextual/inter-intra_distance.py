import argparse
import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse as sp
import pickle
from collections import defaultdict
import sys
sys.path.append('../')
import src.utils as utils

POLYSEMIC_WORDS = {'watch', 'bat', 'virus', 'tank', 'tape', 'bin', 'board',
                   'book', 'bow', 'cap', 'card', 'crane', 'mole',
                   'fan', 'hose', 'keyboard', 'mink', 'mouse',
                   'pipe', 'plug', 'apple'}

class Statistics(object):
    def __init__(self, embedding_path, weighted_intra):
        self.weighted_intra = weighted_intra
        self.embedding_name = ".".join( (os.path.basename(embedding_path).strip().split("."))[0:-1] )
        self.E, self.i2w, self.w2i, self.i2xml, self.xml2i, self.gold_dict_i, self.gold_dict_xml = self.load(embedding_path)
        self.word_cluster_dict, self.word_relevant_size_dict = self.get_clusters()
        self.gather_results()

    def weighted_intra2text(self):
        if self.weighted_intra:
            return "True"
        return "False"

    def load(self, embedding_path):
        i2w = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_i2w.p", 'rb'))
        w2i = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_w2i.p", 'rb'))
        w2i = defaultdict(set)
        for i, w in i2w.items():
            w2i[w].add(i)
        E = sp.load_npz(embedding_path)
        E = E.tocsr()
        i2xml = pickle.load(open('../data/indexing/gold_labels/train/' + self.embedding_name + "_i2xml.p", 'rb'))
        xml2i = pickle.load(open('../data/indexing/gold_labels/train/' + self.embedding_name + "_i2xml.p", 'rb'))
        gold_dict_i = pickle.load(open('../data/indexing/gold_labels/train/' + self.embedding_name + "_gold_dict_i.p", 'rb'))
        gold_dict_xml = pickle.load(open('../data/indexing/gold_labels/train/' + self.embedding_name + "_gold_dict_xml.p", 'rb'))
        return E, i2w, w2i, i2xml, xml2i, gold_dict_i, gold_dict_xml

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

    def get_clusters(self):
        gold_dict_ids = set(self.gold_dict_i.keys())
        word_cluster_dict = defaultdict(set)
        word_relevant_size_dict = defaultdict(int)
        for w, ids in self.w2i.items():
            clusters = defaultdict(set)
            for i in ids:
                if i in gold_dict_ids:
                    word_relevant_size_dict[w] += 1
                    labels = self.gold_dict_i[i]
                    for l in labels:
                        clusters[l].add(i)
            if len(clusters.keys()) > 0:
                word_cluster_dict[w] = clusters
        return word_cluster_dict, word_relevant_size_dict

    def inter_intra_distance(self):
        word_data_dict = {}
        for w, clusters in self.word_cluster_dict.items():
            data = {}
            if len(clusters.keys())> 1 and len(clusters.keys()): # TODO del
                inter_d = self.inter_distance(clusters)
                intra_d = self.intra_distance(clusters)
                if inter_d is None or intra_d is None:
                    continue
                ratio = [intra_d[0]/inter_d[0], intra_d[1]/inter_d[1]]

                if not np.isnan(ratio[0]) and not np.isnan(ratio[1]):
                    data["size"] = self.word_relevant_size_dict[w]
                    data["inter"] = inter_d
                    data["intra"] = intra_d
                    data["ratio"] = ratio
                    word_data_dict[w] = data

        return word_data_dict

    def gather_results(self):
        word_data_dict = self.inter_intra_distance()
        print("Metrics for", self.embedding_name, ":")

        # global nominator + denominator
        global_intra = np.array([0.0, 0.0])
        global_inter = np.array([0.0, 0.0])
        for word, data in word_data_dict.items():
            global_intra += data["intra"]
            global_inter += data["inter"]
        global_ratio = np.array([global_intra[0]/global_inter[0], global_intra[1]/global_inter[1]])
        print("Global ratio (intra/inter): ", global_ratio)

        # weighted average
        weighted_ratio = np.array([0.0, 0.0])
        denom = 0.0
        for word, data in word_data_dict.items():
            weighted_ratio += np.array([data["ratio"][0]*data["size"], data["ratio"][1]*data["size"]])
            denom += data["size"]
        weighted_ratio = np.array([weighted_ratio[0]/denom, weighted_ratio[1]/denom])
        print("Weighted ratio: ", weighted_ratio)

        out_dir = "../results/intra_inter_distance/"
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        out_file = open(out_dir+"intra_inter_distance.csv", "a")
        if os.path.getsize(out_dir+"intra_inter_distance.csv") == 0:
            out_file.write("name\tglobal_jaccard\tglobal_cosine\tweighted_jaccard\tweighted_cosine\tweighted_intra\n")
        text = [self.embedding_name, str(global_ratio[0]), str(global_ratio[1]), str(weighted_ratio[0]),  str(weighted_ratio[1]), self.weighted_intra2text()]
        text = "\t".join(text) + "\n"
        out_file.write(text)

    def intra_distance(self, clusters):
        # TODO exclude zeros in diag when computing mean
        """
        Compute distances within clusters.
        :param clusters: clusters for a word
        :return:  average pairwise distances within clusters
        """
        avg_intra_C = 0
        avg_intra_J = 0
        denom = 0 #len(clusters.keys())
        for c, ids in clusters.items():
            cluster_size = len(ids)
            if cluster_size > 1:
                indices = sorted(ids)
                D = self.E[indices, :]
                assert len(indices) == D.shape[0]
                J = pairwise_distances(D.todense(), metric="jaccard")
                C = pairwise_distances(D.todense(), metric="cosine")
                # compute mean excluding diagonal
                local_denom = 0
                J_nom = 0
                C_nom = 0
                for i in range(J.shape[0]):
                    for j in range(J.shape[0]):
                        if i!= j:
                            local_denom += 1
                            J_nom += J[i, j]
                            C_nom += C[i, j]
                d_J = J_nom/local_denom
                d_C = C_nom/local_denom
                # d_J = np.mean(np.mean(J))
                # d_C = np.mean(np.mean(C))
                if self.weighted_intra:
                    avg_intra_C += d_C*cluster_size
                    avg_intra_J += d_J*cluster_size
                    denom += cluster_size
                else:
                    avg_intra_C += d_C
                    avg_intra_J += d_J
                    denom += 1
        if denom == 0:
            return None
        return np.array([avg_intra_J/denom, avg_intra_C/denom])

    def inter_distance(self, clusters):
        """
        Compute distances among clusters.
        :param clusters: clusters for a word
        :return:  average pairwise distances among clusters
        """
        avg_inter_C = 0
        avg_inter_J = 0
        denom = 0
        D_matrices = []
        for c, ids in clusters.items():
            indices = sorted(ids)
            D = self.E[indices, :]
            D_matrices.append(D)
            assert D.shape[0] > 0
            assert len(indices) == D.shape[0]

        for i in range(len(D_matrices)):
            for j  in range(len(D_matrices)):
                if i!=j:
                    D1 = D_matrices[i]
                    D2 = D_matrices[j]
                    J = pairwise_distances(D1.todense(), D2.todense(), metric="jaccard")
                    C = pairwise_distances(D1.todense(), D2.todense(), metric="cosine")
                    d_J = np.mean(np.mean(J))
                    d_C = np.mean(np.mean(C))
                    avg_inter_C += d_C
                    avg_inter_J += d_J
                    denom += 1

        return np.array([avg_inter_J/denom, avg_inter_C/denom])

def main():
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--embedding', required=False,
                            default='../data/sparse_matrices/word_base/embeddings/2000_0.5_train_alphas.txt.gz.npz', type=str,
                            help='Path to npz format contextual sparse embedding (training).')
        parser.add_argument('--weighted-intra', required=False, default=False, type=bool, help='True if intra distance should be weighted, too. Default: False')
        args = parser.parse_args()
        print("The command line arguments were ", args)

        Statistics(args.embedding, args.weighted_intra)
if __name__ == "__main__":
    main()