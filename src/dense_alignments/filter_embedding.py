import argparse
import os
import numpy as np
import pickle
sys.path.append('../')
import src.steps.utils as utils

FREQUENCY_LIMIT = 400000
class Embedding(object):
    def __init__(self, embedding_path):
        self.embedding_name = ".".join( (os.path.basename(embedding_path).strip().split("."))[0:-1] )
        self.E, self.i2w, self.w2i, self.word_order = self.load(embedding_path)

    def load(self, embedding_path):
        i2w = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_i2w.p", 'rb'))
        w2i = pickle.load(open('../data/indexing/words/embeddings/' + self.embedding_name + "_w2i.p", 'rb'))
        E = pickle.load(open(embedding_path, 'rb'))
        print(type(E))
        word_order = pickle.load(open('../data/word_frequency/' + self.embedding_name + '_frequency.p',  'rb'))
        return E, i2w, w2i, word_order

    def filter(self, vocabulary, name, animals=None):
        row_list = []
        word_list = []
        print('vocab size: ', len(vocabulary))
        if animals != None:
            vocab_set = set(vocabulary).intersection(set(animals))
            vocabulary =  list(vocab_set)
        for word in vocabulary:
            w = word.strip()
            ind = self.w2i.get(w, None)
            if ind != None:
                row_list.append(ind)
                word_list.append(word)

        row_list = sorted(row_list)
        F = self.E[row_list, :]
        w2i = {self.i2w[original_ind]: new_ind for (new_ind, original_ind) in enumerate(row_list)}
        i2w = {ind: word for (word, ind) in w2i.items()}
        local_freq_limit = np.min([FREQUENCY_LIMIT, len(i2w.keys())])
        F = F[0:local_freq_limit, :]
        i2w = {ind: word for (ind, word) in i2w.items() if ind < local_freq_limit}
        w2i = {word: ind for (ind, word) in i2w.items()}

        print("Dense matrix shape: ", F.shape, "\nIndices: ", len(i2w.keys()))
        index_name = "../data/indexing/words/embeddings/" + self.embedding_name + "_f_" + name
        matrix_name = "../data/dense_matrices/word_base/embeddings/filtered/" + self.embedding_name \
                   + "_f_" + name + ".p"
        if animals != None:
            index_name = "../data/indexing/words/embeddings/animals_" + self.embedding_name + "_f_" + name
            matrix_name = "../data/dense_matrices/word_base/embeddings/filtered/animals_" + self.embedding_name \
                       + "_f_" + name + ".p"
        utils.pickler((index_name + "_i2w.p"), i2w)
        utils.pickler((index_name + "_w2i.p"), w2i)
        utils.pickler(matrix_name, F)
        return F, i2w, w2i

def main():
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--embedding', required=True,
                            default='../data/dense_matrices/word_base/embeddings/glove.6B.400k.300d.txt.p', type=str,
                            help='Path to preprocessed dense matrix.')
        parser.add_argument('--vocabulary', required=True,
                            default='../../all/data/vocabulary/conceptnet56_top50000_vocabulary.p', type=str,
                            help='Pickled vocabulary file.')
        # parser.add_argument('--animals', required=False,
        #                     default=None, type=str)
        args = parser.parse_args()
        print("The command line arguments were ", args)

        se = Embedding(args.embedding)
        vocabulary = pickle.load(open(args.vocabulary, 'rb'))
        vocabulary_name = (os.path.basename(args.vocabulary).strip().split("_vocabulary"))[0]

        # animals = pickle.load(open(args.animals, 'rb'))
        print("Global filter")
        F = se.filter(vocabulary, vocabulary_name)


if __name__ == "__main__":
    main()