import argparse
import os
import pickle
import sys
import copy
sys.path.append('../')
import src.utils as utils
import scipy.sparse as sp
from collections import defaultdict

class Dict_merger(object):
    def __init__(self, sparse_path):
        self.embedding_name, self.E = self.load(sparse_path)
        self.d_sim = self.find_dict()

    def load(self, sparse_path):
        embedding_name = os.path.basename(sparse_path)
        E = sp.load_npz(sparse_path)
        return embedding_name, E

    def find_dict(self):
        short_embedding_name = (self.embedding_name.strip().split(".emb_f_"))[0]
        out_name = "../data/dict_similarity/" + short_embedding_name + "_dict_similarity.p"
        d = pickle.load(open(out_name, 'rb'))
        return d

    def merge(self):
        merged_dict = defaultdict(list)
        bases_to_drop = set()
        bases_to_keep = list()
        for base, sim in self.d_sim.items():
            values = [item for sublist in merged_dict.values() for item in sublist]
            if base not in values:
                merged_dict[base] = sim
                bases_to_keep.append(base)
            else:
                bases_to_drop.add(base)

        new_E = copy.deepcopy(self.E)
        for i in range(self.E.shape[1]):
            if i in bases_to_drop:
                similar_bases = [base for base, sim in merged_dict.items() if i in sim]
                for similar_base in similar_bases:
                    new_E[:, similar_base] += self.E[:, i]


        dropped_new_E = new_E[:, bases_to_keep]
        print(dropped_new_E.shape)
        return dropped_new_E

    def save_merged(self):
        new_E = self.merge()
        name = "../data/sparse_matrices/word_base/embedding/dict_merged/" + self.embedding_name
        utils.save_npz(name, new_E)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sparse-matrix', required=False, type=str,
                        default='../data/sparse_matrices/word_base/embeddings/filtered/glove300d_l_0.5.emb_f_propagated_animals_conceptnet56.npz',
                        help='Path to npz format sparse matrix')
    args = parser.parse_args()
    print("Command line arguments were ", args)
    dm = Dict_merger(args.sparse_matrix)
    dm.save_merged()


if __name__ == "__main__":
    main()