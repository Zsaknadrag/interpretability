import argparse
import os
import numpy as np
import pickle
import json
import sys
sys.path.append('../')
import src.utils as utils
from sklearn.preprocessing import normalize
import scipy.sparse as sp
from collections import defaultdict, OrderedDict
import copy

def concept_frequency(concept_name):
    dict_path = "../data/word_concept_dict/" + concept_name + "_word_concept_dict.p"
    word_concept_dict = pickle.load(open(dict_path, 'rb'))
    concept_freq = defaultdict(int)
    for word, clist in word_concept_dict.items():
        for c in clist:
            concept_freq[c] += 1
    return concept_freq

def ban_concept(concept):
    banned_concepts = []  # ["Synonym", "RelatedTo", "IsA"]
    for c in banned_concepts:
        if concept.find(c) != -1:
            return True
    return False

class Concept2Base(object):
    def __init__(self, aligner):
        self.aligner = aligner

    def get_concept_base_pairs(self):
        """
        Associate a base to each concept based on the maximum ppmi value per concept.
        :return:
        c_max_base: dictionary for (concept: base) pairs
        """
        # PPMI = self.calculate_PPMI()
        ppmi_name = os.path.join(("../results/bbd/matrix/" + self.aligner.rand2text()), self.aligner.embedding_name, self.aligner.concept_name,
                                 ("bbd_mtx.npz"))
        PPMI = sp.load_npz(ppmi_name)
        print("BBD matrix loaded...")
        max_bases = np.argmax(PPMI, axis=1)  # max concepts based on ppmi values
        c_max_base = {}
        none_number = 0
        for i in range(max_bases.shape[0]):
            row = PPMI[i, :]
            max_value = np.amax(row)
            j = max_bases[i, 0]
            c_name = self.aligner.i2c[i]
            b_ind = j
            c_max_base[c_name] = b_ind
            if max_value == 0.0:
                none_number += 1
                c_max_base[c_name] = "NONE"
        self.save_max_base(c_max_base)

        return c_max_base

    def save_max_base(self, mtx):
        if self.aligner.longname:
            file_name = os.path.join(("../results/bbd/max_base/" + self.aligner.rand2text()), self.aligner.embedding_name, self.aligner.concept_name,
                                     (
                                     "max_base_of_a_concept_" + self.aligner.embedding_name + "_" + self.aligner.concept_name + "_thd" + str(
                                         self.aligner.thd) + "_" + self.aligner.cont2text() + ".p"))
        else:
            file_name = os.path.join(("../results/bbd/max_base/" + self.aligner.rand2text()), self.aligner.embedding_name, self.aligner.concept_name,
                                     ("max_base_of_a_concept.p"))
        utils.pickler(file_name, mtx)

    def sample_max_base(self):
        c_max_base = self.get_concept_base_pairs()
        if self.aligner.longname:
            f_name = os.path.join(("../results/bbd/max_base/" + self.aligner.rand2text()), self.aligner.embedding_name, self.aligner.concept_name,
                                     (
                                     "max_base_of_a_concept_" + self.aligner.embedding_name + "_" + self.aligner.concept_name + "_thd" + str(
                                         self.aligner.thd) + "_" + self.aligner.cont2text() + ".csv"))
        else:
            f_name = os.path.join(("../results/bbd/max_base/" + self.aligner.rand2text()), self.aligner.embedding_name, self.aligner.concept_name,
                                     ("max_base_of_a_concept.csv"))

        words_dict = {}
        words_dict["NONE"] = ""
        for i in range(self.aligner.E.shape[1]):
            col = ((self.aligner.E.getcol(i)).toarray().T)[0, :]
            ind = np.argpartition(col, -20)[-20:]  # max five indices
            ind[np.argsort(col[ind])]
            words = [self.aligner.i2w[j] for j in ind]
            # words = [(i.strip().split("en:"))[-1] for i in words]
            words = " ".join(words)
            words_dict[i] = words

        f = open(f_name, "w", encoding="utf-8")
        for concept, base in c_max_base.items():
            if str(base) != "NONE":
                out_text = concept + "\t" + str(base) + "\t" + words_dict[base] + "\n"
                f.write(out_text)
        f.close()

class Base2Concept(object):
    def __init__(self, aligner):
        self.aligner = aligner

    def get_base_concept_pairs(self, one2one=True):
        """
        Associate a concept to each base based on the maximum ppmi value per base.
        :return:
        b_max_concept: dictionary for (base: concept) pairs
        """
        b_max_concepts = defaultdict(list)
        # PPMI = self.calculate_PPMI()
        ppmi_name = os.path.join(("../results/bbd/matrix/" + self.aligner.rand2text()), self.aligner.embedding_name, self.aligner.concept_name,
                                 ("bbd_mtx.npz"))
        PPMI = sp.load_npz(ppmi_name)

        if one2one:
            max_cs = np.argmax(PPMI, axis=0)  # get max of each column ie base
            # max_ppmi_values = np.amax(PPMI, axis=0)
            b_max_concept = {}
            for i in range(max_cs.shape[1]):
                col = PPMI[:, i]
                max_value = np.amax(col)
                j = max_cs[0, i]
                c_name = self.aligner.i2c[j]
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
                column = PPMI[:, i]
                max_value = max_values[0, i]
                if max_value == 0.0:
                    b_max_concepts[i] = []
                else:
                    concept_inds = [ind for ind, value in enumerate(column) if value == max_value]
                    for concept_ind in concept_inds:
                        concept_name = self.aligner.i2c[concept_ind]
                        b_max_concepts[i].append(concept_name)

            self.save_max_concept(b_max_concepts, one2one)
        # print("b_max_concepts: ", len(b_max_concepts.keys()))
        return b_max_concepts

    def save_max_concept(self, mtx, one2one=True):
        addon = ""
        if not one2one:
            addon = "s"

        if self.aligner.longname:
            file_name = os.path.join(("../results/bbd/max_concept" + addon + "/" + self.aligner.rand2text()), self.aligner.embedding_name,
                                     self.concept_name,
                                     (
                                     "max_concepts_of_base_" + self.aligner.embedding_name + "_" + self.aligner.concept_name + "_thd" + str(
                                         self.aligner.thd) + "_" + self.aligner.cont2text() + ".p"))
        else:
            file_name = os.path.join(("../results/bbd/max_concept" + addon + "/" + self.aligner.rand2text()), self.aligner.embedding_name,
                                     self.aligner.concept_name,
                                     ("max_concepts_of_base.p"))
        utils.pickler(file_name, mtx)

    def sample_max_concept(self, one2one=True):
        b_max_concept = self.get_base_concept_pairs(one2one=one2one)
        addon = ""
        if not one2one:
            addon = "s"
        if self.aligner.longname:
            f_name = os.path.join(("../results/bbd/max_concept" + addon + "/" + self.aligner.rand2text()), self.aligner.embedding_name,
                                     self.aligner.concept_name,
                                     (
                                     "max_concepts_of_base_" + self.aligner.embedding_name + "_" + self.aligner.concept_name + "_thd" + str(
                                         self.aligner.thd) + "_" + self.aligner.cont2text() + ".csv"))
        else:
            f_name = os.path.join(("../results/bbd/max_concept" + addon + "/" + self.aligner.rand2text()), self.aligner.embedding_name,
                                     self.aligner.concept_name,
                                     ("max_concepts_of_base.csv"))
        f = open(f_name, "w", encoding="utf-8")
        for i in range(self.aligner.E.shape[1]):
            col = ((self.aligner.E.getcol(i)).toarray().T)[0, :]
            nonzero = [j for (j, value) in enumerate(col) if value > 0]
            ind = np.random.choice(nonzero, 5)
            words = [self.aligner.i2w[i] for i in ind]
            words = " ".join(words)
            concepts = b_max_concept[i]
            if type(concepts) == type(list()):
                concepts = "[ "+ " ".join(concepts) + " ]"
            out_text = str(i) + "\t" + concepts + "\t" + words + "\n"
            f.write(out_text)
        f.close()

class Aligner(object):
    def __init__(self, embedding_path, concept_path, cont_sparse, thd, language, longname):
        self.longname = longname
        self.thd = thd
        self.cont_sparse = cont_sparse
        self.no_random_embedding_name, self.embedding_name, self.concept_name, self.random = self.format_name(
            embedding_path, concept_path)
        self.E, self.C = self.load_matrices(embedding_path, concept_path)
        print("init_parameters:\nembedding ", self.E.shape, "\nconcept ", self.C.shape)
        self.i2c, self.c2i, self.i2w, self.w2i, self.word_concept_dict = self.load_files()

    def cont2text(self):
        if self.cont_sparse:
            return "continous"
        else:
            return "binary"

    def rand2text(self):
        if self.random:
            return "random1/"
        else:
            return ""

    def format_name(self, embedding_path, concept_path):
        embedding_name = ".".join((os.path.basename(embedding_path).strip().split("."))[0:-1])
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
        mtx_name = os.path.join(("../data/sparse_matrices/word_concept/weighted/" + addon), self.embedding_name,
                                self.concept_name,
                                ("weighted_word_concept_matrix.npz"))
        C = sp.load_npz(mtx_name)
        return E, C

    def load_files(self):
        i2c = pickle.load(open(('../data/indexing/concept/' + self.concept_name + "_i2c.p"), 'rb'))
        c2i = pickle.load(open('../data/indexing/concept/' + self.concept_name + "_c2i.p", 'rb'))
        i2w = pickle.load(open('../data/indexing/words/embeddings/' + self.no_random_embedding_name + "_i2w.p", 'rb'))
        w2i = pickle.load(open('../data/indexing/words/embeddings/' + self.no_random_embedding_name + "_w2i.p", 'rb'))
        word_concept_dict = pickle.load(
            open(('../data/word_concept_dict/' + self.concept_name + "_word_concept_dict.p"), 'rb'))
        # vocab = pickle.load( open(('../data/vocabulary/'+ self.concept_name + "_list.p"), 'rb') )
        return i2c, c2i, i2w, w2i, word_concept_dict

    def preproc_C(self):
        # denseC = self.C.todense()
        C = sp.lil_matrix(self.C)
        concept_freq = concept_frequency(self.concept_name)
        for j in range(C.shape[1]):
            concept = self.i2c[j]
            if concept_freq[concept] < self.thd or ban_concept(concept):
                C[:, j] = 0.0
        return sp.csr_matrix(C)

    def product(self):
        """
                Compute the product of transposed concept matrix and sparse word embedding matrix,
                so that the result matrix has its rows represent concepts while its columns represent bases.
                :return:
                concept_base_mtx: product matrix
                """
        # concept_word_mtx = self.C.transpose()

        concept_word_mtx = self.preproc_C().transpose()
        word_base_binary = self.E
        if not self.cont_sparse:
            word_base_binary = (word_base_binary > 0).astype(np.int_)
        concept_base_mtx = concept_word_mtx * word_base_binary
        print("\nproduct: ", concept_base_mtx.shape)

        ret = concept_base_mtx  # self.postproc_product(concept_base_mtx)
        self.save_product(concept_base_mtx)
        return ret

    def postproc_product(self, mtx):
        concept_freq = concept_frequency(self.concept_name)
        mtx = sp.lil_matrix(mtx)
        mtx[mtx < self.thd] = 0
        return sp.csr_matrix(mtx)

    def save_product(self, mtx):
        concept_base_mtx_name = os.path.join(("../data/sparse_matrices/concept_base/" + self.rand2text()),
                                             self.embedding_name, self.concept_name, str(self.thd),
                                             ("concept_base_mtx" + "_" + self.cont2text() + ".npz"))
        utils.save_npz(concept_base_mtx_name, mtx)

    def calculate_BBD(self):
        # context_base_mtx_name = "../data/sparse_matrices/concept_base/concept_base_mtx_" + self.raw_type + "_1.npz"
        # CB_matrix = sp.load_npz(context_base_mtx_name)
        # WB_mtx = normalize(self.E, axis=0, norm='l1')
        # WC_mtx = normalize(self.C, axis=0, norm='l1')
        WB_mtx = copy.deepcopy(self.E)
        WC_mtx = copy.deepcopy(self.C)

        WB_mtx.data = np.exp(WB_mtx.data)
        WC_mtx.data = np.exp(WC_mtx.data)
        c_freq = WC_mtx.sum(axis=0)  # row vector
        b_freq = WB_mtx.sum(axis=0)
        WB_mtx = WB_mtx/b_freq
        WC_mtx = WC_mtx/c_freq


        WB_mtx.data =np.sqrt(WB_mtx.data)
        WC_mtx.data = np.sqrt(WC_mtx.data)
        cooccurrences = sp.csr_matrix(WC_mtx.transpose().dot(WB_mtx))
        cooccurrences.data = - np.log(cooccurrences.data)
        cooccurrences.data = np.nan_to_num(cooccurrences.data)
        # cooccurrences[cooccurrences < 0] = 0
        print("BBD: ", cooccurrences.shape)
        self.save_BBD(cooccurrences)
        return cooccurrences

    def save_BBD(self, mtx):
        ppmi_name = os.path.join(("../results/bbd/matrix/" + self.rand2text()), self.embedding_name, self.concept_name,
                                 ("bbd_mtx.npz"))
        utils.save_npz(ppmi_name, mtx)

    def get_alignments(self):
        b2c = Base2Concept(self)
        b2c.sample_max_concept()
        b2c.sample_max_concept(one2one=False)
        c2b = Concept2Base(self)
        c2b.sample_max_base()

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sparse-matrix', required=False, type=str,
                        default='../data/sparse_matrices/word_base/embeddings/glove100d_l_01emb.npz',
                        help='Path to npz format sparse matrix')
    parser.add_argument('--concept-file', required=False, type=str,
                        default='../data/conceptnet/assertions/conceptnet56_assertions.json',
                        help='Path to short ConceptNet json file')
    parser.add_argument('--language', required=False, type=str, default='en',
                        help='Language of conceptnet and sparse matrix files. Default: en')
    parser.add_argument('--cont-sparse', required=False, type=bool, default=False,
                        help='True if sparse embedding should be continous instead of binary. Default: False')
    parser.add_argument('--thd', required=False, type=int, default=20,
                        help='Treshold for concept frequency. Default: 5')
    parser.add_argument('--longname', required=False, type=bool, default=False,
                        help='')
    args = parser.parse_args()
    print("Command line arguments were ", args)
    # cm = Aligner(args.sparse_matrix, args.concept_file, args.cont_sparse, args.thd, args.language)

    for thd in [40, 30, 20, 10, 5]:
        cm = Aligner(args.sparse_matrix, args.concept_file, args.cont_sparse, thd, args.language, args.longname)
        print("CALCULATING ", thd, " threshold...")
        cm.calculate_BBD()
        cm.get_alignments()


if __name__ == "__main__":
    main()
