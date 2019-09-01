import argparse
import os
import numpy as np
import pickle
import sys
sys.path.append('../')
import src.utils as utils
import ot
import scipy.sparse as sp
import scipy.spatial.distance as dist
from collections import defaultdict

class Aligner(object):
    def __init__(self, embedding_path, concept_path, thd, language, longname):
        self.longname = longname
        self.thd = thd
        self.no_random_embedding_name, self.embedding_name, self.concept_name, self.random = self.format_name(embedding_path, concept_path)
        self.E, self.C = self.load_matrices(embedding_path, concept_path)
        self.i2c, self.c2i, self.i2w, self.w2i, self.word_concept_dict = self.load_files()

    def format_name(self, embedding_path, concept_path):
        embedding_name = ".".join( (os.path.basename(embedding_path).strip().split("."))[0:-1] )
        concept_name = (os.path.basename(concept_path).strip().split("_assertions"))[0] + "_t" + str(self.thd)

        random = False
        if embedding_path.find("random1") != -1:
            random = True
        no_random_embedding_name = embedding_name
        if random:
            paths = os.path.split(embedding_path)
            base = (((paths[1].strip().split("_random"))[0]).split(".npz"))[0]
            no_random_embedding_name = base
        return no_random_embedding_name, embedding_name, concept_name, random

    def load_matrices(self, embedding_path, concept_path):
        addon = ""
        if self.random:
            addon = "random1/"
        E = sp.load_npz(embedding_path)
        mtx_name = os.path.join(("../data/sparse_matrices/word_concept/" + addon), self.embedding_name, self.concept_name,
                                  ("word_concept_mtx.npz"))
        C = sp.load_npz(mtx_name)
        return E, C

    def load_files(self):
        i2c = pickle.load( open(('../data/indexing/concept/'+ self.concept_name + "_i2c.p"), 'rb') )
        c2i = pickle.load( open('../data/indexing/concept/'+ self.concept_name + "_c2i.p", 'rb') )
        i2w = pickle.load( open('../data/indexing/words/embeddings/'+ self.no_random_embedding_name + "_i2w.p", 'rb') )
        w2i = pickle.load( open('../data/indexing/words/embeddings/' + self.no_random_embedding_name + "_w2i.p", 'rb') )
        word_concept_dict = pickle.load( open(('../data/word_concept_dict/'+ self.concept_name + "_word_concept_dict.p"), 'rb') )
        # vocab = pickle.load( open(('../data/vocabulary/'+ self.concept_name + "_list.p"), 'rb') )
        return i2c, c2i, i2w, w2i, word_concept_dict

    def calculate_gromov_wasserstein(self):
        for i in range(self.E.shape[1]):
            # get ith col of E
            col = ((self.E.getcol(i)).toarray().T)[0, :]
            col = col.reshape(len(col), 1)
            embedding_col = col[col > 0]
            for j in range(self.C.shape[1]):
                concept_col0 = (((self.C.getcol(j)).toarray().T)[0, :]).astype(np.bool)
                concept_col0 = concept_col0.reshape(len(concept_col0), 1)
                concept_col = concept_col0[col>0]
                concept_col = concept_col.reshape(len(concept_col),1)

                inverted_concept_col0 = np.array((concept_col0 <= 0).astype(np.bool))
                inverted_concept_col0 = inverted_concept_col0.reshape(len(inverted_concept_col0),1)
                inverted_concept_col = inverted_concept_col0[col>0]
                inverted_concept_col = inverted_concept_col.reshape(len(inverted_concept_col),1)

                C1 = dist.cdist(concept_col, concept_col)
                C2 = dist.cdist(inverted_concept_col, inverted_concept_col)
                with np.errstate(divide='ignore'):
                    C1 /= C1.max()
                    C2 /= C2.max()
                    C1[C1 == np.inf] = 0
                    C2[C2 == np.inf] = 0
                    print(C1.shape, C2.shape)
                    concept_col = concept_col.reshape(concept_col.shape[0])
                    inverted_concept_col = inverted_concept_col.reshape(inverted_concept_col.shape[0])
                    print(concept_col.shape, inverted_concept_col.shape)
                    gw0, log0 = ot.gromov.gromov_wasserstein(
                        C1, C2, concept_col, inverted_concept_col, 'square_loss', verbose=True, log=True)
                    print('Gromov-Wasserstein distances: ' + str(log0['gw_dist']))
        gw = 0
        gw = sp.csr_matrix(gw)
        self.save_GW(gw)
        return gw

    def save_GW(self, mtx):
        addon = ""
        if self.random:
            addon = "random1/"
        ppmi_name = os.path.join(("../data/Bhattaccharyya/sparse_matrices/bbd/"+addon), self.embedding_name, self.concept_name, str(self.thd), ("bbd_mtx_" + self.cont2text() + ".npz"))
        utils.save_npz(ppmi_name, mtx)

    def get_concept_base_pairs(self):
        """
        Associate a base to each concept based on the maximum ppmi value per concept.
        :return:
        c_max_base: dictionary for (concept: base) pairs
        """
        PPMI = self.calculate_BBD()
        max_bases = np.argmax(PPMI, axis=1)  # max concepts based on ppmi values
        c_max_base = {}
        none_number = 0
        for i in range(max_bases.shape[0]):
            row = PPMI[i, :]
            max_value = np.amax(row)
            j = max_bases[i,0]
            c_name = self.i2c[i]
            b_ind = j
            c_max_base[c_name] = b_ind
            if max_value == 0.0:
                none_number += 1
                c_max_base[c_name] = "NONE"
        self.save_max_base(c_max_base)

        return c_max_base

    def save_max_base(self, mtx):
        addon = ""
        if self.random:
            addon = "random1/"
        file_name = os.path.join(("../data/Bhattaccharyya/max_base/"+addon), self.embedding_name, self.concept_name, str(self.thd),
                                 ("max_base_of_a_concept_" + self.embedding_name + "_" + self.concept_name + "_thd" + str(self.thd) + "_" +  self.cont2text() + ".p"))
        utils.pickler(file_name, mtx)

    def sample_max_base(self):
        addon = ""
        if self.random:
            addon = "random1/"
        c_max_base = self.get_concept_base_pairs()
        f_name = os.path.join(("../data/Bhattaccharyya/max_base/"+addon), self.embedding_name, self.concept_name, str(self.thd),
                              ("max_base_of_a_concept_" + self.embedding_name + "_" + self.concept_name + "_thd" + str(self.thd) + "_" +  self.cont2text() + ".csv") )

        words_dict = {}
        words_dict["NONE"]=""
        for i in range(self.E.shape[1]):
            col = ((self.E.getcol(i)).toarray().T)[0,:]
            ind = np.argpartition(col, -20)[-20:]  # max five indices
            ind[np.argsort(col[ind])]
            words = [self.i2w[j] for j in ind]
            # words = [(i.strip().split("en:"))[-1] for i in words]
            words = " ".join(words)
            words_dict[i] = words

        f = open(f_name, "w", encoding="utf-8")
        for concept, base in c_max_base.items():
            if str(base) != "NONE":
                out_text = concept + "\t" + str(base) + "\t" + words_dict[base] + "\n"
                f.write(out_text)
        f.close()

    def get_base_concept_pairs(self, one2one = True):
        """
        Associate a concept to each base based on the maximum ppmi value per base.
        :return:
        b_max_concept: dictionary for (base: concept) pairs
        """
        b_max_concepts = defaultdict(list)
        PPMI = self.calculate_BBD()
        if one2one:
            max_cs = np.argmax(PPMI, axis=0)  # get max of each column ie base
            # print(max_cs[0,:])
            max_ppmi_values = np.amax(PPMI, axis=0)
            # max_cs2 = max_cs[0]
            b_max_concept = {}
            for i in range(max_cs.shape[1]):
                col = PPMI[:, i]
                max_value = np.amax(col)
                j = max_cs[0, i]
                c_name = self.i2c[j]
                b_ind = i
                # print(i, c_name)
                b_max_concept[b_ind] = c_name
                if max_value == 0.0:
                    b_max_concept[b_ind] = "NONE"

            b_max_concepts = b_max_concept
            self.save_max_concept(b_max_concepts, one2one)

        else:
            max_values = np.amax(PPMI, axis=0).todense()
            for i in range(PPMI.shape[1]):
                column = PPMI[:,i]
                max_value = max_values[0,i]
                if max_value == 0.0:
                    b_max_concepts[i] = []
                else:
                    concept_inds = [ind for ind,value in enumerate(column) if value == max_value]
                    for concept_ind in concept_inds:
                        concept_name = self.i2c[concept_ind]
                        b_max_concepts[i].append(concept_name)

            self.save_max_concept(b_max_concepts, one2one)
        return b_max_concepts

    def save_max_concept(self, mtx, one2one = True):
        addon = ""
        if not one2one:
            addon = "s"
        addon2 = ""
        if self.random:
            addon2 = "random1/"
        file_name = os.path.join(("../data/Bhattaccharyya/max_concept" + addon + "/" + addon2), self.embedding_name, self.concept_name, str(self.thd),
                                 ("max_concepts_of_base_" + self.embedding_name + "_" + self.concept_name + "_thd" + str(self.thd) + "_" + self.cont2text() + ".p"))
        utils.pickler(file_name, mtx)

    def sample_max_concept(self, one2one=True):
        b_max_concept = self.get_base_concept_pairs(one2one=one2one)
        addon = ""
        if not one2one:
            addon = "s"

        addon2 = ""
        if self.random:
            addon2 = "random1/"
        f_name = os.path.join(("../data/Bhattaccharyya/max_concept" + addon + "/" + addon2), self.embedding_name, self.concept_name, str(self.thd),
                              ("max_concepts_top5_words_" + self.embedding_name + "_" + self.concept_name + "_thd" + str(self.thd) + "_" + self.cont2text() + ".csv") )
        f = open(f_name, "w", encoding="utf-8")
        for i in range(self.E.shape[1]):
            # col = W[:, i]
            col = ((self.E.getcol(i)).toarray().T)[0,:]
            # col = col.toarray().T
            # col = col[0, :]
            nonzero = [j for (j, value) in enumerate(col) if value > 0]
            ind = np.random.choice(nonzero, 5)

            # ind = np.argpartition(col, -5)[-5:]  # max five indices
            # ind[np.argsort(col[ind])]
            words = [self.i2w[i] for i in ind]
            # words = [(i.strip().split("en:"))[-1] for i in words]
            words = " ".join(words)
            concepts = b_max_concept[i]
            if type(concepts) == type(list()):
                concepts = " ".join(concepts)
            out_text = str(i) + "\t" + concepts + "\t" + words + "\n"
            f.write(out_text)
        f.close()

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sparse-matrix', required=False, type=str, default='../data/sparse_matrices/word_base/embeddings/glove50d_reduced_animals_l_0.5.emb.npz', help='Path to npz format sparse matrix')
    parser.add_argument('--concept-file', required=False, type=str, default='../data/conceptnet/assertions/propagated_animals_conceptnet56_assertions.json', help='Path to short ConceptNet json file')
    parser.add_argument('--language', required=False, type=str, default='en', help='Language of conceptnet and sparse matrix files. Default: en')
    parser.add_argument('--thd', required=False, type=int, default=40, help='Treshold for concept frequency. Default: 40')
    parser.add_argument('--longname', required=False, type=bool, default=False, help='Default: False')

    args = parser.parse_args()
    print("Command line arguments were ", args)
    cm = Aligner(args.sparse_matrix, args.concept_file, args.thd, args.language, args.longname)
    cm.calculate_gromov_wasserstein()
    cm.sample_max_concept()
    # cm.sample_max_concept(one2one=False)
    cm.sample_max_base()

if __name__ == "__main__":
     main()