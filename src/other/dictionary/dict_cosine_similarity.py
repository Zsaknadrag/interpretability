import os
import sys
sys.path.append('../')
import src.utils as utils
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class Dict_sim(object):
    def __init__(self, dict_path, thd):
        self.embedding_name = (os.path.basename(dict_path).strip().split(".dict"))[0]
        self.d = sp.load_npz(dict_path)
        self.thd = thd

    def cosine_similarity(self):
        similarity_dict = defaultdict(list)
        similarities = cosine_similarity(self.d.transpose())
        for i in range(similarities.shape[0]):
            col = enumerate(similarities[:, i])
            similar = [ind for ind, sim in col if (sim >= self.thd and ind != i)]
            similarity_dict[i] = similar
        # print(similarity_dict)
        out_name = "../data/dict_similarity/" + self.embedding_name + "_dict_similarity.p"
        utils.pickler(out_name, similarity_dict)



def main():
    THD = 0.5
    DIR =  "../data/sparse_matrices/word_base/dict/"
    dir_list = os.listdir(DIR)
    print(dir_list)
    DICT_PATH = "../data/sparse_matrices/word_base/dict/glove100d_l_0.1.dict.npz"
    for file in dir_list:
        full_path = os.path.join(DIR, file)
        ds = Dict_sim(full_path, THD)
        ds.cosine_similarity()


if __name__ == "__main__":
    main()
