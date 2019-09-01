import os
import json
import argparse
import pickle

OUTDIR = '../data/'

class McRae_filter(object):
    def __init__(self, vocab_path, assertion_path):
        self.vocabulary = pickle.load(open(vocab_path, 'rb'))
        self.assertion = json.load(open(assertion_path, encoding='utf-8'))
        self.name = (os.path.basename(assertion_path).strip().split('_assertions'))[0] \
                    + '_v_' + (os.path.basename(vocab_path).strip().split('_vocabulary'))[0] + '_assertions'

    def filter(self):
        to_keep = []
        for data in self.assertion:
            start = (data['start'].strip().split('/'))[-1]
            if start in self.vocabulary:
                to_keep.append(data)
        print(len(to_keep))
        self.save(to_keep)

    def save(self, data_list):
        dumped = json.dumps(data_list)
        out_json = open(OUTDIR + 'assertions/' + self.name + '.json', 'w')
        print('Output saved to ', OUTDIR + 'assertions/' + self.name + '.json')
        out_json.write(dumped)
        out_json.close()

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--assertion', required=False, type=str,
                        default='../data/assertions/conceptnet56_assertions.json', help='Path to ConceptNet csv.')
    parser.add_argument('--vocabulary', required=False, type=str,
                        default='../data/vocabulary/CONCS_FEATS_concstats_brm.txt_np_vocabulary.p')

    args = parser.parse_args()
    print("The command line arguments were ", args)
    p = McRae_filter(args.vocabulary, args.assertion)
    p.filter()

if __name__ == "__main__":
    main()
