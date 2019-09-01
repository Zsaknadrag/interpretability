import os
import pickle
import sys
sys.path.append('../')
import src.utils as utils
import spacy
import string
nlp = spacy.load('en_core_web_sm')

def load_i2w(embedding_name):
    indices_dir = "../data/indexing/words/embeddings/"
    w2i_path = indices_dir + embedding_name + "_i2w.p"
    i2w = pickle.load(open(w2i_path, 'rb'))
    i2w = sorted(i2w.items(), key=lambda e: int(e[0]))
    return i2w

def filter_words(word_list):
    filtered_list = []

    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    banned_words = ["'s", "\'\'", "``", "--"]
    banned_words.extend(spacy_stopwords)
    banned_words.extend(string.punctuation)

    for word in word_list:
        if word not in banned_words:# and not word.isdigit():
            filtered_list.append(word)
    return filtered_list

def get_words(i2w, microsoft_words):
    word_list = []
    for i,w in i2w:
        word_list.append(w)
    word_list = filter_words(word_list)
    print(word_list[0:30])
    microsoft_word_set = set(word_list).intersection((set(microsoft_words)))
    microsoft_word_list = [word for word in word_list if word in microsoft_word_set]
    print(microsoft_word_list[0:30])
    return word_list, microsoft_word_list

def save_words(words_list, out_path):
    utils.makedir(out_path)
    out_file = open(out_path, "w", encoding="utf-8")
    out_file.write(words_list[0])
    for i in range(1, len(words_list)):
        out_file.write("\n" + words_list[i])
    utils.pickler((out_path + ".p"), words_list)

def write_word_list(file_name, out_dir, microsoft_vocabulary):
    i2w = load_i2w(file_name)
    words, microsoft_words = get_words(i2w, microsoft_vocabulary)
    print("len: ", len(words), len(microsoft_words))
    out_path = out_dir + file_name + "_words"
    save_words(words, out_path)
    out_path_microsoft = out_dir + file_name + "_microsoft_words"
    save_words(microsoft_words, out_path_microsoft)

def main():
    out_dir = "../results/coherence/words/"
    in_dir = "../data/sparse_matrices/word_base/embeddings/"

    microsoft_vocabulary = pickle.load(open("../data/vocabulary/microsoft_concept_graph.json_vocabulary.p", "rb"))
    in_dir_list = os.listdir(in_dir)
    for file_name in in_dir_list:
        full_path = in_dir + file_name
        if os.path.isfile(full_path) and file_name.find("_f_") == -1 and file_name.find("emb") != -1:
            file_name = ".".join((file_name.strip().split("."))[0:-1])
            print(file_name)
            write_word_list(file_name, out_dir, microsoft_vocabulary)


if __name__ == "__main__":
    main()