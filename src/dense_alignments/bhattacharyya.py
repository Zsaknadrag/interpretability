import argparse
import os
import numpy as np
import pickle
sys.path.append('../')
import src.steps.utils as utils
import scipy.sparse as sp
from collections import defaultdict, OrderedDict
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.neighbors.kde import KernelDensity

class Bhattacharyya(object):
    def __init__(self, embedding_path, concept_path, concept_indices_dir, thd):
        self.thd = thd
        self.embedding_name, self.concept_name = self.format_name(embedding_path)
        self.E, self.ES, self.C, self.C_i2w, self.C_w2i = self.load_matrices(embedding_path, concept_path, concept_indices_dir)
        print("init_parameters:\nembedding ", self.E.shape, "\nconcept ", self.C.shape)
        self.i2c, self.c2i, self.i2w, self.w2i, self.word_concept_dict = self.load_files()

    def format_name(self, embedding_path):
        embedding_name = ".".join((os.path.basename(embedding_path).strip().split("."))[0:-1])
        assert embedding_name.find("_f_") != -1
        concept_name = (embedding_name.strip().split("_f_"))[-1] + "_t" + str(self.thd)
        return embedding_name, concept_name

    def load_matrices(self, embedding_path, concept_path, concept_indices_dir):
        E = pickle.load(open(embedding_path, 'rb'))
        mtx_name = os.path.join(("../data/sparse_matrices/word_concept/splitted/" ), self.embedding_name,
                                self.concept_name,
                                ("word_concept_mtx.npz"))
        scaler = StandardScaler()
        ES = scaler.fit_transform(E)
        E = normalize(E, norm='l2', axis=1)
        C = sp.load_npz(concept_path)
        C = normalize(C, norm='l2', axis=1)
        C = sp.csc_matrix(C)
        for file in os.listdir(concept_indices_dir):
            if file.find("i2w") != -1:
                path = os.path.join(concept_indices_dir, file)
                C_i2w = pickle.load(open(path, 'rb'))
            elif file.find("w2i") != -1:
                path = os.path.join(concept_indices_dir, file)
                C_w2i = pickle.load(open(path, 'rb'))

        return E, ES, C, C_i2w, C_w2i

    def load_files(self):
        i2c = pickle.load(open(('../data/indexing/concept/' + self.concept_name + "_i2c.p"), 'rb'))
        c2i = pickle.load(open('../data/indexing/concept/' + self.concept_name + "_c2i.p", 'rb'))
        i2w = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_i2w.p", 'rb'))
        w2i = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_w2i.p", 'rb'))
        word_concept_dict = pickle.load(
            open(('../data/word_concept_dict/' + self.concept_name + "_word_concept_dict.p"), 'rb'))
        return i2c, c2i, i2w, w2i, word_concept_dict

    def transform_indices_C2E(self, indices):
        w2i_words = set(self.w2i.keys())
        words = set([self.C_i2w[i] for i in indices])
        out_inds = [self.w2i[w] for w in words if w in w2i_words]
        return out_inds

    def distributions(self):
        distr = {}
        for i in range(298, self.E.shape[1]):
            for j in range(self.C.shape[1]):
                in_category = self.transform_indices_C2E(self.C.getcol(j).nonzero()[0])
                out_category = self.transform_indices_C2E([k for k in range(self.C.shape[0]) if k not in set(in_category)])
                # print(out_category)
                X_p = np.matrix([self.E[in_category, i]]).T
                X_q = np.matrix([self.E[out_category, i]]).T

                p = KernelDensity(kernel='gaussian').fit(X_p)
                q = KernelDensity(kernel='gaussian').fit(X_q)
                distr[(i, j)] = (p, q)
        distr_name = "../data/bhattacharyya/distr/test/" + self.embedding_name + "/" + str(self.thd) + "_bhattacharyya_distr.p"
        utils.pickler(distr_name, distr)
        return distr

    def distance(self):
        distr_name = "../data/bhattacharyya/distr/test/" + self.embedding_name + "/" + str(self.thd) + "_bhattacharyya_distr.p"
        distr = pickle.load(open(distr_name, "rb"))
        B = np.zeros((self.E.shape[1], self.C.shape[1]))
        for i in range(298, self.E.shape[1]):
            for j in range(self.C.shape[1]):
                p = distr[(i, j)][0]
                q = distr[(i, j)][1]
                samples = np.linspace(-1, 1, 10)[:, np.newaxis]
                log_dens_p = p.score_samples(samples)
                log_dens_q = q.score_samples(samples)
                prod = np.multiply(log_dens_p, log_dens_q)
                summ = np.sum(np.sqrt(prod))
                d = - np.log(summ)
                B[i, j] = d
                print(i, j, d)
        B_name = "../data/bhattacharyya/distance/test/" + self.embedding_name + "/" + str(self.thd) + "_bhattacharyya_distances.p"
        utils.pickler(B_name, B)
        return B

    def analyze(self, col_ind=299):
        B_name = "../data/bhattacharyya/distance/test/" + self.embedding_name + "/" + str(self.thd) + "_bhattacharyya_distances.p"
        B = pickle.load(open(B_name, 'rb'))
        print(B.shape)
        print(self.ES.shape)
        col = B[col_ind, :]
        amax = np.argmax(col)
        vmax = np.max(col)
        print("Max: ", vmax, self.i2c[amax])
        Ecol = self.ES[col_ind, :]
        Esorted = sorted(enumerate(Ecol), reverse=True, key=lambda t: float(t[0]))[0:20]
        words = [self.i2w[i] for i, v in Esorted]

        # print(col.shape)
        # for i in range(len(col)):
        #     print(i, col[i], self.i2c[i])
        # P = np.matmul(self.ES, B)
        # print(P.shape)
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--embedding', required=False, type=str,
                        default='../data/dense_matrices/word_base/embeddings/filtered/glove.6B.400k.300d.txt_f_conceptnet56_top50000.p',
                        help='Path to npz format sparse matrix')
    parser.add_argument('--concept', required=False, type=str,
                        default='../data/sparse_matrices/word_concept/splitted/conceptnet56_top50000_t40/word_concept_mtx.npz',
                        help='Word concept matrix in npz format')
    parser.add_argument('--concept-indices', required=False, type=str,
                        default='../data/indexing/words/embeddings/test/',
                        help='Word concept matrix indices directory, containing i2w and w2i pickled files.')
    parser.add_argument('--thd', required=False, type=int, default=40,
                        help='Treshold for concept frequency. Default: 40')
    parser.add_argument('--longname', required=False, type=bool, default=False,
                        help='')
    args = parser.parse_args()
    print("Command line arguments were ", args)
    # cm = Aligner(args.sparse_matrix, args.concept_file, args.cont_sparse, args.thd, args.language, args.longname, args.rel)
    # cm.calculate_NPPMI()
    # cm.get_alignments()
    # cm.max_max_alignment()
    for thd in [40]:#[40, 30, 20, 10, 5]:#0
        cm = Bhattacharyya(args.embedding, args.concept, args.concept_indices, thd)
        # print("Computing distributions...")
        # cm.distributions()
        # print("Computing distances...")
        # distr = cm.distance()
        print("Analysis...")
        cm.analyze()



if __name__ == "__main__":
    main()
