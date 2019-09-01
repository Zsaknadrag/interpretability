import argparse
import os
import numpy as np
import pandas as pd
sys.path.append('../../')
import src.utils

class DenseEmbedding(object):
    """
    This class provides utils for efficient storage and manipulation of dense (embedding) matrices.
    Objects are assumed to be located in the rows.
    """

    def __init__(self, embedding_path, matrix_name=None, languages=None, filter_rows=-1):
        if matrix_name == None:
            matrix_name = os.path.basename(embedding_path)
            print(matrix_name)
        self.w2i, self.i2w, self.W = self.load_embeddings(embedding_path, matrix_name, languages=languages)

    def load_embeddings(self, path, mtx_name, languages=None, filter_rows=-1):
        """
        Reads in the dense embedding file.
        Parameters
        ----------
        path : location of the gzipped sparse embedding file
        languages : a set containing the languages to filter for.
        If None, no filtering takes plce.
        filter_rows : indicates the number of lines to read in.
        If negative, the entire file gets processed.
        Returns
        -------
        w2i : wordform to identifier dictionary
        i2w : identifier to wordform dictionary
        W : the sparse embedding matrix
        """

        if type(languages) == str:
            languages = set([languages])
        elif type(languages) == list:
            languages = set(languages)

        df = pd.read_csv(path, sep=" ", quoting=3, header=None, index_col=0)
        glove = {key: val.values for key, val in df.T.items()}
        words = list(glove.keys())
        vectors = list(glove.values())
        i2w = {i:w for i, w in enumerate(words)}

        W = np.matrix(vectors)
        w2i = {w: i for i, w in i2w.items()}
        frequency_ranking = list(i2w.values())

        # save dense matrix
        mtx_path_name = os.path.join("../data/dense_matrices/word_base/embeddings/", (mtx_name+'.p'))
        utils.pickler(mtx_path_name, W)

        # save w2i, i2w
        w2i_name = "../data/indexing/words/embeddings/" + mtx_name + "_w2i.p"
        i2w_name = "../data/indexing/words/embeddings/" + mtx_name + "_i2w.p"
        frequency_name = "../data/word_frequency/" + mtx_name + "_frequency.p"
        print('keys list: ', len(i2w.keys()))
        print('values set: ', len(set(i2w.values())))
        utils.pickler(w2i_name, w2i)
        utils.pickler(i2w_name, i2w)
        utils.pickler(frequency_name, frequency_ranking)
        return w2i, i2w, W

    def get_embedding(self, word, language=''):
        query_word = '{}{}{}'.format(language, '' if len(language) == 0 else ':', word)
        return self.W.getrow(self.w2i[query_word])


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--embedding', required=True, type=str)
    parser.add_argument('--matrix-name', required=False, type=str)
    parser.add_argument('--language', required=False, default=None, nargs='*', type=str)
    # parser.add_argument('--vocabulary', required=False, default='../data/vocabulary/microsoft_concept_graph_w_10.json_vocabulary.p', type=str)
    args = parser.parse_args()
    print("The command line arguments were ", args)
    se = DenseEmbedding(args.embedding, args.matrix_name, args.language)

    # print('{} words read in...'.format(se.W.shape[0]))


if __name__ == "__main__":
    main()
