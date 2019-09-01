import argparse
import os
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
import src.utils as utils
import seaborn as sns
import scipy.sparse as sp
from collections import defaultdict


class Inspector(object):
    def __init__(self, sparse_path):
        self.sparse_name = self.format_name(sparse_path)
        self.S = self.load_files(sparse_path)
        self.i2w, self.w2i = self.load_indices()

    def format_name(self, concept_path):
        base_name = os.path.basename(concept_path)
        sparse_name = ".".join((base_name.strip().split('.'))[0:-1])
        print(sparse_name)
        return sparse_name

    def load_files(self, sparse_path):
        C = sp.load_npz(sparse_path)
        print("Sparse matrix shape: ", C.shape)
        return C

    def load_indices(self):
        i2w = pickle.load(open(('../data/indexing/words/embeddings/' + self.sparse_name + "_i2w.p"), 'rb'))
        w2i = pickle.load(open('../data/indexing/words/embeddings/' + self.sparse_name + "_w2i.p", 'rb'))
        return i2w, w2i

    def calculate_cooccurrence(self):
        BB = self.S.T * self.S
        print("Cooccurence shape: ", BB.shape)
        return BB

    def calculate_NPPMI(self):
        cooccurrences = self.calculate_cooccurrence()
        fs = self.S.sum(axis=0)
        denom = fs.T @ fs
        pmi = sp.csr_matrix( (self.S.shape[0] * cooccurrences) / denom )
        pmi.data = np.log(pmi.data)
        prod = sp.csr_matrix(cooccurrences / float(self.S.shape[0]))
        prod.data = 1 / - np.log(prod.data)
        nppmi = pmi.multiply(prod)
        nppmi.data = np.nan_to_num(nppmi.data)
        # nppmi[nppmi < 0] = 0.0
        nppmi[nppmi < 0.0 ] = 0.0
        sub = nppmi.diagonal()*np.eye(nppmi.shape[0])

        nppmi = nppmi - sub
        nppmi = sp.csr_matrix(nppmi)

        return nppmi

    def analyze_NPPMI(self):
        nppmi = sp.csc_matrix(self.calculate_NPPMI())
        close_concepts_dict = defaultdict(set)
        counter = 0
        # max_values = np.amax(nppmi, axis=0).todense()
        max_nppmi = nppmi.max(axis=0).max()
        nnz_list = []
        for i in range(nppmi.shape[0]):
            nnz_number = nppmi.getcol(i).count_nonzero()
            nnz_list.append(nnz_number)


        out_dir = "../results/close_bases/img/"
        utils.makedir(out_dir)
        self.plot_nnz_frequency(nnz_list, self.sparse_name)
        self.plot_nppmi_mtx(nppmi, self.sparse_name)
        self.plot_nppmi_values(nppmi, self.sparse_name)

        # print("Number of bases having close neighbours: ", counter)
        # print("Max ppmi found: ", max_nppmi)
        out_name = "../results/close_bases/" + self.sparse_name + "_nonzero_number.p"
        utils.pickler(out_name, close_concepts_dict)

    def plot_nppmi_mtx(self, nppmi, title):
        fig = plt.figure()
        nppmi = nppmi.todense()
        sns.heatmap(nppmi, vmin=0, vmax=1)
        plt.title(title)
        plt.xlabel("Bases")
        plt.ylabel("Bases")
        plt.yticks([])
        plt.xticks([])
        plt.show()
        plt.tight_layout()
        out_name = "../results/close_bases/img/" + self.sparse_name + "_heatmap.png"
        fig.savefig(out_name)

    def plot_nppmi_values(self, nppmi, title):
        fig = plt.figure()
        data = nppmi.data
        plt.hist(data, rwidth=0.95, log=True, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.ylim([0, 100000])
        plt.title(title)
        plt.ylabel("Frequency")
        plt.xlabel("Nppmi value")
        plt.show()
        plt.tight_layout()
        out_name = "../results/close_bases/img/" + self.sparse_name + "_nppmi_hist.png"
        fig.savefig(out_name)

    def plot_nnz_frequency(self, nnz_list, title):
        fig = plt.figure()
        plt.hist(nnz_list, rwidth=0.95, bins=range(0, 801, 100))
        plt.ylim([0, 400])
        plt.title(title)
        plt.ylabel("Frequency")
        plt.xlabel("Number of close (non-zero nppmi) bases")
        plt.show()
        out_name = "../results/close_bases/img/" + self.sparse_name + "_nnz_hist.png"
        plt.tight_layout()
        fig.savefig(out_name)

    def get_words(self, base1, base2):
        coll = ((self.S.getcol(base1)).toarray().T)[0, :]
        nonzero1 = [(j, value) for (j, value) in enumerate(coll) if value > 0]
        nonzero1 = sorted(nonzero1, key=lambda e: e[1], reverse=True)
        words1 = [self.i2w[i] for i,v in nonzero1]

        col2 = ((self.S.getcol(base2)).toarray().T)[0, :]
        nonzero2 = [(j, value) for (j, value) in enumerate(col2) if value > 0]
        nonzero2 = sorted(nonzero2, key=lambda e: e[1], reverse=True)
        words2 = [self.i2w[i] for i,v in nonzero2]

        print(base1, ": ", words1[0:10])
        print(base2, ": ", words2[0:10])
        intersection = list(set(words1).intersection(set(words2)))
        print("intersection: ", len(intersection), "\n", intersection[0:20], "\n")

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sparse',
                        required=True,
                        type=str,
                        default='../data/sparse_matrices/word_base/embeddings/glove300d_l_0.5_DL_top400000.emb.gz.npz',
                        help='Path to npz format sparse embedding matrix')

    args = parser.parse_args()
    print("Command line arguments were ", args)
    insp = Inspector(args.sparse)
    insp.analyze_NPPMI()

if __name__ == "__main__":
     main()