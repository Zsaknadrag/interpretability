import os
import argparse
import pickle
import copy
import numpy as np
import scipy.sparse as sp
from collections import defaultdict, OrderedDict
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import src.steps.utils as utils

from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import TfidfModel

class topic_model(object):
    def __init__(self, embedding_path, thd, topics):
        self.num_topics = topics
        self.thd = thd
        self.no_random_embedding_name, self.embedding_name, self.random, self.concept_name = self.format_name(embedding_path)
        self.E = self.load_matrices(embedding_path)
        self.i2w, self.w2i, self.i2c, self.c2i, self.word_concept_dict = self.load_files()


    def format_name(self, embedding_path):
        embedding_name = ".".join((os.path.basename(embedding_path).strip().split("."))[0:-1])

        random = False
        if embedding_path.find("random1") != -1:
            random = True
        no_random_embedding_name = embedding_name
        if random:
            paths = os.path.split(embedding_path)
            base = (((paths[1].strip().split("_random"))[0]).split(".npz"))[0]
            no_random_embedding_name = base

        concept_name = (embedding_name.strip().split("_f_"))[-1]  + "_t" + str(self.thd)

        return no_random_embedding_name, embedding_name, random, concept_name

    def load_matrices(self, embedding_path):
        addon = ""
        if self.random:
            addon = "random1/"
        E = sp.load_npz(embedding_path)
        return E

    def load_files(self):
        i2w = pickle.load(open('../data/indexing/words/embeddings/' + self.no_random_embedding_name + "_i2w.p", 'rb'))
        w2i = pickle.load(open('../data/indexing/words/embeddings/' + self.no_random_embedding_name + "_w2i.p", 'rb'))
        i2c = pickle.load(open(('../data/indexing/concept/' + self.concept_name + "_i2c.p"), 'rb'))
        c2i = pickle.load(open('../data/indexing/concept/' + self.concept_name + "_c2i.p", 'rb'))
        word_concept_dict = pickle.load(
            open(('../data/word_concept_dict/' + self.concept_name + "_word_concept_dict.p"), 'rb'))
        return i2w, w2i, i2c, c2i, word_concept_dict

    def proportionate_E(self):
        # embedding_mtx = copy.deepcopy(self.E)
        # embedding_mtx.data = np.exp(embedding_mtx.data)
        # b_freq = embedding_mtx.sum(axis=0)
        # embedding_mtx = sp.csr_matrix(embedding_mtx / b_freq)

        embedding_mtx = normalize(self.E, norm='l2', axis=1)
        D = np.eye(self.E.shape[1]) * self.E.shape[1]
        embedding_mtx = sp.csr_matrix(np.ceil(embedding_mtx * D))
        print("embedding_mtx: ", embedding_mtx.shape)
        return embedding_mtx

    def make_text_data(self):
        embedding_mtx = self.proportionate_E()
        text_data = []
        for j in range(embedding_mtx.shape[1]):
            tokens = []
            for i in range(embedding_mtx.shape[0]):
                t = [self.i2w[i] for k in range(int(embedding_mtx[i,j]))]
                tokens.extend(t)
            text_data.append(tokens)
        print("\tSample text data: ", text_data[0:5])
        text_data_name = "../results/topic_model/text_data/" + self.embedding_name + "_text_data.p"
        utils.pickler(text_data_name, text_data)
        # print(text_data[1])
        return text_data

    def preproc(self):
        text_data_name = "../results/topic_model/text_data/" + self.embedding_name + "_text_data.p"
        text_data = pickle.load(open(text_data_name, 'rb'))

        words = self.w2i.keys()
        print("words: ", len(words))
        dct = Dictionary(text_data)
        dct_name = "../results/topic_model/dct/" + self.embedding_name + "_dct.p"
        if not os.path.exists(os.path.basename(dct_name)):
            os.makedirs(os.path.basename(dct_name))
        utils.pickler(dct_name, dct)
        print("dict: ", len(dct))

        corpus = [dct.doc2bow(text) for text in text_data]
        corpus_name = "../results/topic_model/corpus/" + self.embedding_name + "_corpus.p"
        utils.pickler(corpus_name, corpus)

        tfidf = TfidfModel(corpus)
        tfidf_corpus = tfidf[corpus]
        tfidf_corpus_name = "../results/topic_model/corpus/" + self.embedding_name + "_tfidf_corpus.p"
        utils.pickler(tfidf_corpus_name, tfidf_corpus)
        return dct, corpus, tfidf_corpus

    def lda_model(self):
        passes = 20
        dct_name = "../results/topic_model/dct/" + self.embedding_name + "_dct.p"
        corpus_name = "../results/topic_model/corpus/" + self.embedding_name + "_corpus.p"
        tfidf_corpus_name = "../results/topic_model/corpus/" + self.embedding_name + "_tfidf_corpus.p"
        dct = pickle.load(open(dct_name, 'rb'))
        corpus = pickle.load(open(corpus_name, 'rb'))
        tfidf_corpus = pickle.load(open(tfidf_corpus_name, 'rb'))

        lda_model = LdaMulticore(corpus, num_topics=self.num_topics, id2word=dct,
                                 passes=passes, workers=4, alpha='symmetric', decay=0.6)
        lda_model_name = "../results/topic_model/lda_model/" + self.embedding_name + "/" + self.embedding_name + "_" + str(self.num_topics) + "_model.gensim"
        utils.makedir(lda_model_name)
        lda_model.save(lda_model_name)
        #
        # topics = lda_model.print_topics(num_words=10)
        # for topic in topics:
        #     print(topic)
        tfidf_lda_model = LdaMulticore(tfidf_corpus, num_topics=self.num_topics, id2word=dct,
                                       passes=passes, workers=4, alpha='symmetric', decay=0.6)
        tfidf_lda_model_name = "../results/topic_model/lda_model/" + self.embedding_name + "/" + self.embedding_name + "_" + str(self.num_topics) + "_tfidf_model.gensim"
        utils.makedir(tfidf_lda_model_name)
        tfidf_lda_model.save(tfidf_lda_model_name)
        print("\tLDA model saved...")

    def dominant_words(self, base, number=10):
        col = ((self.E.getcol(base)).toarray().T)[0, :]
        nonzero = [self.i2w[ind] for ind, val in sorted(enumerate(col), reverse=True, key=lambda t: float(t[1]))
                   if val > 0][0:number]
        return nonzero

    def topics_per_doc(self):
        texts_name = "../results/topic_model/text_data/" + self.embedding_name + "_text_data.p"
        dct_name = "../results/topic_model/dct/" + self.embedding_name + "_dct.p"
        dictionary = pickle.load(open(dct_name, 'rb'))
        texts = pickle.load(open(texts_name, 'rb'))

        model_name = "../results/topic_model/lda_model/" + self.embedding_name + "/" + self.embedding_name + "_" + str(
            self.num_topics) + "_model.gensim"
        model = LdaModel.load(model_name)
        # docs = model.get_document_topics()
        file_name = "../results/topic_model/topics_per_document/" + self.embedding_name + "/" + str(self.num_topics) + "_docs.csv"
        utils.makedir(file_name)
        out_file = open(file_name, 'w', encoding='utf-8')
        out_file.write("base\tnumber_of_topics\ttopics\n")
        for i in range(len(texts)):
            out_file.write( str(i) )
            # print(i, "th base")
            # print(self.random_words(i))
            bow = dictionary.doc2bow(texts[i])
            doc = model.get_document_topics(bow)
            topics = []
            for ind, p in doc:
                t = str(model.show_topic(ind))
                t = " ".join(t.split(","))
                topics.append(t)
                # print("\t", model.show_topic(ind))

            topics_str = "\t".join(topics)
            out_file.write(("\t" + str(len(topics)) + "\t"))
            out_file.write((topics_str + "\n"))
            # print('\n',str(i), "\t", str(len(topics)), topics_str)

    def get_topics(self):
        model_name = "../results/topic_model/lda_model/" + self.embedding_name + "/" + self.embedding_name + "_" + str(
            self.num_topics) + "_model.gensim"
        if os.path.exists(model_name):
            print("Loading model ", model_name)
            model = LdaModel.load(model_name)
        else:
            return
            print("Computing model...( ", str(self.num_topics) + ")\n", model_name)
            model = self.lda_model()
        topics = model.print_topics(self.num_topics, 10)

        file_name = "../results/topic_model/topics/" + self.embedding_name + "/" + str(self.num_topics) + "_topics.csv"
        utils.makedir(file_name)
        out_file = open(file_name, 'w', encoding='utf-8')
        for t in topics:
            text = str(t).split(",")
            text = " ".join(text)
            out_file.write((str(text)+"\n"))

        return topics

    def get_model(self):
        model_name = "../results/topic_model/lda_model/" + self.embedding_name + "/" + self.embedding_name + "_" + str(
            self.num_topics) + "_model.gensim"
        if os.path.exists(model_name):
            print("Loading model ", model_name)
            model = LdaModel.load(model_name)
        else:
            return
            print("Computing model...( ", str(self.num_topics) + ")\n", model_name)
            model = self.lda_model()
        return model

    def compute_coherence_values(self, start=10, limit=100, step=10):
        coherence_values = []
        texts_name = "../results/topic_model/text_data/" + self.embedding_name + "_text_data.p"
        dct_name = "../results/topic_model/dct/" + self.embedding_name + "_dct.p"
        corpus_name = "../results/topic_model/corpus/" + self.embedding_name + "_corpus.p"
        tfidf_corpus_name = "../results/topic_model/corpus/" + self.embedding_name + "_tfidf_corpus.p"

        dictionary = pickle.load(open(dct_name, 'rb'))
        corpus = pickle.load(open(corpus_name, 'rb'))
        tfidf_corpus = pickle.load(open(tfidf_corpus_name, 'rb'))
        texts = pickle.load(open(texts_name, 'rb'))
        for num_topics in range(start, limit+1, step):
            model = LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes = 15, workers=4, alpha='symmetric')
            model_name = "../results/topic_model/lda_model/" + self.embedding_name + "/" + self.embedding_name + "_" + str(
                num_topics) + "_model.gensim"
            utils.makedir(model_name)
            model.save(model_name)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
            print(num_topics)
            print(coherencemodel.get_coherence())
        cv_name = "../results/topic_model/coherence_values/" + self.embedding_name + "_from" + str(start) + "_to" + str(limit) + ".p"
        utils.pickler(cv_name, coherence_values)
        return coherence_values

    def plot_coherence_values(self, start, limit, step):
        # coherence_values = self.compute_coherence_values(start, limit, step)
        cv_name = "../results/topic_model/coherence_values/" + self.embedding_name + "_from" + str(start) + "_to" + str(limit) + ".p"
        coherence_values = pickle.load(open(cv_name, 'rb'))
        print(coherence_values)
        x = range(start, limit+1, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.tight_layout()
        # plt.show()
        out_dir = "../results/topic_model/coherence_values/"
        utils.makedir(out_dir)
        plt.savefig((out_dir + "from" + str(start) + "_to" + str(limit) + "_cv.png"))

    def concept_frequency(self):
        word_concept_dict = self.word_concept_dict
        concept_freq = defaultdict(int)
        for word, clist in word_concept_dict.items():
            for c in clist:
                concept_freq[c] += 1
        return concept_freq

    def train_test_concepts(self, test_size=0.4):
        concept_words_dict = defaultdict(list)
        for word, c_list in self.word_concept_dict.items():
            for concept in c_list:
                concept_words_dict[concept].append(word)
        splitted_cw = defaultdict(list)
        for concept, word_list in concept_words_dict.items():
            test = np.random.choice(word_list, int(np.floor(test_size*len(word_list))))
            train = [word for word in word_list if word not in test]
            splitted_cw[concept] = (train, test)


        train_word_concept_dict = defaultdict(list)
        test_word_concept_dict = defaultdict(list)
        for c, data in splitted_cw.items():
            for w in data[0]:
                train_word_concept_dict[w].append(c)
            for w in data[1]:
                test_word_concept_dict[w].append(c)
        return train_word_concept_dict, test_word_concept_dict, splitted_cw

    def topic_align(self):
        # ----- load files -----
        train_wc_dict, test_wc_dict, splitted_cw = self.train_test_concepts()
        concept_freq = self.concept_frequency()
        frequency_name = os.path.join("../data/word_frequency",
                                      ((self.embedding_name.strip().split("emb"))[0] + "emb_frequency.p"))
        word_freq = pickle.load(open(frequency_name, "rb"))

        model = self.get_model()
        topics = model.show_topics(formatted=False, num_topics=self.num_topics, num_words=20)
        topic_alignment = defaultdict(list)
        file_name = "../results/topic_model/topics_extended/" + self.embedding_name + "/" + str(self.num_topics) + "_topics.csv"
        utils.makedir(file_name)
        out_file = open(file_name, 'w', encoding='utf-8')
        out_file.write("topicID\tWords in topic\tConcepts connected to words\n")

        # ------ compute relevant topics -----
        for t in topics:
            ind = t[0]
            topic_data = t[1]
            # print(ind, topic_data)
            concept_freq_dict = defaultdict(float)
            words = []
            topic_data = sorted(topic_data, reverse=True, key=lambda e: float(e[1]))
            for pair in topic_data:
                word = pair[0]
                value = pair[1]
                concept_list = train_wc_dict[word]
                words.append(word)
                for c in concept_list:
                    concept_freq_dict[c] += float(value)

            concept_freq_dict = {concept: (freq/float(concept_freq[concept])) for (concept, freq) in concept_freq_dict.items()}
            concept_freq_dict = OrderedDict(sorted(concept_freq_dict.items(), key=lambda e: float(e[1]), reverse=True))
            if len(concept_freq_dict.values()) > 0:
                max_concepts = sorted(concept_freq_dict.items(), key=lambda e: float(e[1]), reverse=True)[0:5]
                topic_alignment[ind] = [c for c, val in max_concepts]
                max_concepts = [(concept + " (" + str(value) + ")") for concept, value in max_concepts]
                # print(max_concepts)
                # print("\n")
            else:
                topic_alignment[ind] = []
                max_concepts = []

            max_concepts_to_write = "[ " + " ".join(sorted(max_concepts)) + " ]"
            words_to_write = "[ " + " ".join(words) + " ]"
            out_file.write((str(ind) + "\t" + words_to_write + "\t" + max_concepts_to_write + "\n"))

        return topic_alignment, test_wc_dict, splitted_cw

    def topic_distance(self):
        model = self.get_model()
        topics = model.show_topics(formatted=False, num_topics=self.num_topics, num_words=20)
        for t in topics:
            ind = t[0]
            topic_data = t[1]
            # print(ind, topic_data)
            concept_freq_dict = defaultdict(float)
            words = []
            topic_data = sorted(topic_data, reverse=True, key=lambda e: float(e[1]))
            for pair in topic_data:
                word = pair[0]
                words.append(word)
            # for word in words:
                # get distances from other words



    def document_alignment(self):
        topic_alignment, test_wc_dict, splitted_cw = self.topic_align()

        texts_name = "../results/topic_model/text_data/" + self.embedding_name + "_text_data.p"
        dct_name = "../results/topic_model/dct/" + self.embedding_name + "_dct.p"
        dictionary = pickle.load(open(dct_name, 'rb'))
        texts = pickle.load(open(texts_name, 'rb'))

        model_name = "../results/topic_model/lda_model/" + self.embedding_name + "/" + self.embedding_name + "_" + str(
            self.num_topics) + "_model.gensim"
        model = LdaModel.load(model_name)
        doc_alignment = defaultdict(list)
        for i in range(len(texts)):
            print(i, "th base")
            print(self.dominant_words(i))
            bow = dictionary.doc2bow(texts[i])
            doc = model.get_document_topics(bow)
            for topic_ind, topic_prob in doc:
                concepts = topic_alignment[topic_ind]
                doc_alignment[i].extend(concepts)
                concepts = ", ".join(concepts)
                print("\t", topic_ind, ": ", concepts)

            # topics_str = "\t".join(topics)
            # print('\n',str(i), "\t", str(len(topics)), topics_str)
        return doc_alignment, test_wc_dict, splitted_cw

    def test_set_evaluate(self):
        doc_alignment, test_wc_dict, splitted_cw = self.document_alignment()
        avg_acc = 0.0
        for i in range(self.E.shape[1]):
            col = enumerate(((self.E.getcol(i)).toarray().T)[0, :])
            nonzero_words = [self.i2w[ind] for ind, val in col if val > 0]
            concepts = set(doc_alignment[i])
            test = []
            for c in concepts:
                test.extend(splitted_cw[c][1])
                # test.extend(splitted_cw[c][0])
            # print(test)
            test_size = len(set(test))
            acc = 0.0
            for word in nonzero_words:
                word_concepts = set(self.word_concept_dict.get(word, []))
                # word_concepts = set(test_wc_dict.get(word, []))
                intersection = word_concepts.intersection(concepts)
                if len(intersection) > 0:
                    acc += 1.0
            acc = acc/test_size
            avg_acc += acc
        avg_acc = avg_acc/self.E.shape[1]
        print("average accuracy: ", avg_acc)
        return avg_acc

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--sparse-matrix', required=False, type=str, default='../data/sparse_matrices/word_base/embeddings/filtered/glove300d_l_0.5.emb_f_microsoft_concept_graph_w_10.json.npz', help='Path to npz format sparse matrix')
    parser.add_argument('--thd', required=False, type=int, default=40, help='Treshold for concept frequency. Default: 40')
    parser.add_argument('--topics', required=False, type=int, default=100, help='Number of topics. Default: 20')
    args = parser.parse_args()
    print("Command line arguments were ", args)

    # topics = range(10, 101, 10)
    # for t in topics:
    #     # print("TOPIC number: ", t)
    #     tm = topic_model(args.sparse_matrix, args.thd, t)
    #     tm.get_topics()
    #     # tm.topics_per_doc()
    tm = topic_model(args.sparse_matrix, args.thd, args.topics)
    print("Making text data...")
    tm.make_text_data()
    print("Making corpus, dict...")
    tm.preproc()
    print("Computing LDA model...")
    tm.lda_model()
    tm.test_set_evaluate()
    tm.document_alignment()
if __name__ == "__main__":
     main()