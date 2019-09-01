import os
import numpy as np
import scipy.sparse as sp
import pickle
from collections import OrderedDict
import json
import sys
sys.path.append('../')
import src.utils as utils
import spacy
nlp = spacy.load('en_core_web_sm')

def load_files(e_path, i_path, i_no_s_path):
    # load sparse matrix
    E = sp.csc_matrix(sp.load_npz(e_path))
    embedding_name = ".".join((os.path.basename(e_path).strip().split("."))[0:-1])
    # load indices
    i2w_path = "../data/indexing/words/embeddings/" + embedding_name + "_i2w.p"
    i2w = pickle.load(open(i2w_path, "rb"), encoding='utf-8')
    w2i_path = "../data/indexing/words/embeddings/" + embedding_name + "_w2i.p"
    w2i = pickle.load(open(w2i_path, "rb"), encoding='utf-8')
    # load ConceptNet intersection words
    intersection = open(i_path, "r", encoding="utf-8")
    interseciton_no_s = open(i_no_s_path, "r", encoding="utf-8")
    return E, i2w, w2i, intersection, interseciton_no_s, embedding_name

def get_intersection_words(intersection):
    intersection_words = []
    for line in intersection:
        data = json.loads(line)
        word = data["word"]
        intersection_words.append(word)
    intersection.close()
    return intersection_words

def get_top_bot_words(E, i2w, w2i, i_words, i_no_s_words, out_dir):
    utils.makedir(out_dir)
    out_file = open(out_dir+"top_bot_words.jsons", "w")
    out_file_no_s = open(out_dir + "top_bot_words_no_s.jsons", "w")
    intersected_words = set(i_words).intersection(set(w2i.keys()))
    intersected_words_no_s = set(i_no_s_words).intersection(set(w2i.keys()))
    print("int len: ", len(intersected_words))
    print("int_no_s len: ", len(intersected_words_no_s))
    np.random.seed(1)
    for i in range(E.shape[1]):
        data = OrderedDict()
        data_no_s = OrderedDict()
        col = enumerate((E.getcol(i).toarray().T)[0, :])

        # bottom words
        zero_words = {i2w[ind] for ind, value in col if value == 0}
        zero_words_int = zero_words.intersection(intersected_words)
        zero_words_int_no_s = zero_words.intersection(intersected_words_no_s)
        bot_words = [word for word in i_words if word in zero_words_int]
        bot_words_no_s = [word for word in i_no_s_words if word in zero_words_int_no_s]
        bot_words = np.random.choice(bot_words, 10).tolist()
        bot_words_no_s = np.random.choice(bot_words_no_s, 10).tolist()

        # top words
        col = enumerate((E.getcol(i).toarray().T)[0, :])
        nonzero_words = sorted([(i2w[ind], value) for ind, value in col if value > 0], reverse=True, key=lambda e: float(e[1]))
        top_words = [word for word, value in nonzero_words if word in intersected_words][0:10]
        top_words_no_s = [word for word, value in nonzero_words if word in intersected_words_no_s][0:10]

        # json data
        data["base_id"] = i
        data["top_words"] = top_words
        data["bot_words"] = bot_words

        data_no_s["base_id"] = i
        # data_no_s["top_words"] = top_words_no_s
        data_no_s["bot_words"] = bot_words_no_s

        # write

        write_data(data, out_file)
        write_data(data_no_s, out_file_no_s)
    out_file.close()
    print('Saving file to ', out_dir)
    out_file_no_s.close()

def write_data(data, out_file):
    out_file.write(json.dumps(data)+"\n")

def main():
    out_dir = "../results/coherence/top_words_glove/"
    embedding_path = "../data/sparse_matrices/word_base/embeddings/glove300d_l_0.5_kmeans_top400000.emb.gz.npz"
    intersection_path =  "../results/coherence/topK.jsons"
    interseciton_no_s_path =  "../results/coherence/topK.jsons"
    E, i2w, w2i, intersection, interseciton_no_s, embedding_name = load_files(embedding_path, intersection_path, interseciton_no_s_path)
    intersection_words = get_intersection_words(intersection)
    interseciton_no_s_words = get_intersection_words(interseciton_no_s)
    out_dir = out_dir + embedding_name + "/"
    get_top_bot_words(E, i2w, w2i, intersection_words, interseciton_no_s_words, out_dir)

if __name__ == "__main__":
    main()

