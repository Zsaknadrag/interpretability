import argparse
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pickle
from collections import defaultdict
import sys
sys.path.append('../')
import src.steps.utils as utils

POLYSEMIC_WORDS = {'watch', 'bat', 'virus', 'tank', 'tape', 'bin', 'board',
                   'book', 'bow', 'cap', 'card', 'crane', 'mole',
                   'fan', 'hose', 'keyboard', 'mink', 'mouse',
                   'pipe', 'plug', 'apple'}

class Statistics(object):
    def __init__(self, embedding_path):
        self.embedding_name = ".".join( (os.path.basename(embedding_path).strip().split("."))[0:-1] )
        self.E, self.i2w, self.w2i = self.load(embedding_path)
        # self.get_stats()
        self.get_distribution()

    def load(self, embedding_path):
        i2w = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_i2w.p", 'rb'))
        w2i = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_w2i.p", 'rb'))
        w2i = defaultdict(set)
        for i, w in i2w.items():
            w2i[w].add(i)
        E = sp.load_npz(embedding_path)
        E = E.tocsr()
        return E, i2w, w2i

    def plot_distribution(self, word, r):
        plt.hist(list(np.nditer(r)))
        plt.title(word)
        plt.show()

    def get_distribution(self):
        recurring = self.get_recurring_words()
        for word, inds in recurring.items():
            indices = sorted(inds)
            if len(indices) < 3000 and word in POLYSEMIC_WORDS:
                D = self.E.tocsr()[indices, :]
                print(word, D.shape)
                nonzero_indices = np.nonzero(D)
                columns_non_unique = nonzero_indices[1]
                unique_columns = sorted(set(columns_non_unique))
                E = D.tocsc()[:, unique_columns]
                print(E.shape)
                E = binarize(E)
                r = np.sum(E, axis=0)/len(indices)
                # self.plot_distribution(word, r)
                nonzeros = np.nonzero(E)
                # print(nonzeros[1])
                for col, value in enumerate(np.nditer(r)):

                    if value < 1.0:
                        ii = np.where(nonzeros[1] == col)[0]
                        rows = nonzeros[0][ii]
                        active_ids = [indices[i] for i in rows]
                        # print("active: ", active_ids)
                        if len(active_ids) > 1:
                            print("\nBase: ", col)
                            for ai in active_ids:
                                print(self.get_context(ai))

    def get_stats(self):
        recurring = self.get_recurring_words()
        dir_name = "../results/similarity/" + self.embedding_name + "/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        cosine_file = open(os.path.join(dir_name, "all_cosine_similarity.csv"), "w")
        cos_text = ["word", "#forms", "cosine_similarity", "var", "min", "context1", "context2"]
        cos_text = "\t".join(cos_text)+"\n"
        cosine_file.write(cos_text)

        jacc_file = open(os.path.join(dir_name, "all_jaccard_similarity.csv"), "w")
        jacc_text = ["word", "#forms", "jaccard_similarity", "var", "min", "context1", "context2"]
        jacc_text = "\t".join(jacc_text)+"\n"
        jacc_file.write(jacc_text)

        self.get_similarity(recurring, cosine_file, jacc_file)
        # self.get_intersection(recurring)


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

    def get_similarity(self, dict, cos_file, jacc_file):
        """
        Get pairwise cosine similarity of a set of embeddings corresponding to one (embedded) word.
        :return:
        """
        for word, inds in dict.items():
            indices = sorted(inds)
            if len(indices) < 3000:# and word in POLYSEMIC_WORDS:
                # print(word, len(indices))
                D = self.E.tocsr()[indices, :]
                S = cosine_similarity(D)
                J = 1 - pairwise_distances(D.todense(), metric="jaccard")
                cos_sim = np.mean(np.mean(S, axis=1))
                cos_var = np.var(S)
                cos_min = np.min(np.min(S, axis=1))

                jaccard_sim = np.mean(np.mean(J, axis=1))
                jaccard_min = np.min(np.min(J, axis=1))
                jaccard_var = np.var(J)

                cc1, cc2 = self.get_context_min(S, indices)
                cos_text = [word, str(len(indices)), str(cos_sim), str(cos_var), str(cos_min),cc1, cc2]#, jaccard_sim, jaccard_var, jaccard_min]
                cos_text = "\t".join(cos_text)
                cos_file.write(cos_text + "\n")

                jc1, jc2 = self.get_context_min(J, indices)
                jacc_text = [word, str(len(indices)), str(jaccard_sim), str(jaccard_var), str(jaccard_min), jc1, jc2]#, jaccard_sim, jaccard_var, jaccard_min]
                jacc_text = "\t".join(jacc_text)
                jacc_file.write(jacc_text + "\n")

    def get_context(self, ind, context_size=10):
        c = []
        for i in range(ind - context_size, ind + context_size + 1):
            if i >= 0 and i < len(self.i2w.keys()):
                # print(self.i2w[i], end=" ")
                c.append(self.i2w[i])
        c = " ".join(c)
        return c

    def get_context_min(self, M, indices):
        # get context
        ri, ci = M.argmin() // M.shape[1], M.argmin() % M.shape[1]
        ind1 = indices[ri]
        ind2 = indices[ci]
        context_size = 10
        c1 = []
        # print('\t', end="")
        for i in range(ind1 - context_size, ind1 + context_size + 1):
            if i >= 0 and i < len(self.i2w.keys()):
                # print(self.i2w[i], end=" ")
                c1.append(self.i2w[i])
        # print('\n\t', end="")
        c2 = []
        for i in range(ind2 - context_size, ind2 + context_size + 1):
            if i >= 0 and i < len(self.i2w.keys()):
                # print(self.i2w[i], end=" ")
                c2.append(self.i2w[i])
        # print('\n')
        c1 = " ".join(c1)
        c2 = " ".join(c2)
        return c1, c2

def main():
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--embedding', required=False,
                            default='../data/sparse_matrices/word_base/embeddings/2000_0.1_train_alphas.txt.gz.npz', type=str,
                            help='Path to np format contextual sparse embedding (training).')
        args = parser.parse_args()
        print("The command line arguments were ", args)

        Statistics(args.embedding)
if __name__ == "__main__":
    main()