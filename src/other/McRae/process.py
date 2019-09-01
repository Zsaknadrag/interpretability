import argparse
import os
import json
import sys
from nltk.corpus import stopwords
from collections import defaultdict, OrderedDict

sys.path.append('../')
import src.utils as utils


STOPWORDS = set(stopwords.words('english'))
OUTDIR = '../data/'
class Processor(object):
    def __init__(self, path, polysemy):
        self.name = os.path.basename(path)
        self.polysemy = polysemy
        self.file = open(path, 'r')
        self.preprocess()

    def preprocess(self):
        polysemic_words = set()
        word_concept_dict = defaultdict(set)
        vocabulary = set()
        concepts = set()
        counter = 0
        js_list = []
        for line in self.file:
            if counter != 0:
                parts = line.strip().split('\t')
                word = parts[0]
                concept = parts[1]
                concepts.add(concept)
                if not self.polysemy:
                    word = (word.strip().split('_'))[0]
                    word_concept_dict[word].add(concept)
                    vocabulary.add(word)
                    data = OrderedDict()
                    data['start'] = word
                    data['end'] = concept
                    data['rel'] = 'McRae'
                    data['weight'] = parts[6]
                    data['dataset'] = 'McRae'
                    js_list.append(data)
                else:
                    if word.find('_') == -1:
                        word_concept_dict[word].add(concept)
                        vocabulary.add(word)
                        data = OrderedDict()
                        data['start'] = word
                        data['end'] = concept
                        data['rel'] = 'McRae'
                        data['weight'] = parts[6]
                        data['dataset'] = 'McRae'
                        js_list.append(data)
                    else:
                        polysemic_words.add((word.strip().split('_'))[0])
            counter += 1
        self.save(vocabulary, word_concept_dict, js_list)
        self.get_indices(concepts)
        print("Polysemic Words: ", len(polysemic_words))

    def get_indices(self, concepts):
        i2c = {ind: concept for ind, concept in enumerate(sorted(concepts))}
        c2i = {concept: ind for ind, concept in i2c.items()}
        # print(i2c)
        addon = ''
        if self.polysemy:
            addon = '_np'
        utils.pickler(OUTDIR+'indexing/concept/'+self.name+addon+'_t0_i2c.p', i2c)
        utils.pickler(OUTDIR+'indexing/concept/'+self.name+addon+'_t0_c2i.p', c2i)
        utils.pickler(OUTDIR+'indexing/concept/'+self.name+addon+'_i2c.p', i2c)
        utils.pickler(OUTDIR+'indexing/concept/'+self.name+addon+'_c2i.p', c2i)

    # save files
    def save(self, vocabulary, word_concept_dict, js_list):
        addon = ''
        if self.polysemy:
            addon = '_np'
        utils.pickler(OUTDIR+'vocabulary/'+self.name+addon+'_t0_vocabulary.p', vocabulary)
        utils.pickler(OUTDIR+'word_concept_dict/'+self.name+addon+'_t0_word_concept_dict.p', word_concept_dict)
        utils.pickler(OUTDIR+'vocabulary/'+self.name+addon+'_vocabulary.p', vocabulary)
        utils.pickler(OUTDIR+'word_concept_dict/'+self.name+addon+'_word_concept_dict.p', word_concept_dict)

        dumped = json.dumps(js_list)
        out_json = open(OUTDIR+'assertions/'+self.name+addon+'.json', 'w')
        out_json.write(dumped)
        out_json.close()

        print("Lenght of vocabulary: ", len(vocabulary))

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', required=False, type=str,
                        default='../data/McRae/McRae-BRM-InPress/CONCS_FEATS_concstats_brm.txt', help='Path to ConceptNet csv.')
    parser.add_argument('--polysemy', required=False, type=bool,
                        default=False)

    args = parser.parse_args()
    print("The command line arguments were ", args)
    p = Processor(args.path, args.polysemy)


if __name__ == "__main__":
    main()
