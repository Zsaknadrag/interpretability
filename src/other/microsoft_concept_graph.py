import os
import json
import pickle
from collections import OrderedDict

"""
Create assertion file from microsoft concept graph. Relations are named "IsA".
"""

IN_PATH = 'C:/Users/Vanda/Downloads/data-concept/data-concept/data-concept-instance-relations.txt'
OUT_PATH = '../data/conceptnet/assertions/microsoft_concept_graph_w_2.json'
REL_TYPE = '/r/IsA'
DATASET = 'microsoft_concept_graph'
W2I_PATH = '../data/indexing/words/embeddings/glove300d_l_0.1.emb_w2i.p'

in_file = open(IN_PATH, 'r')
out_file = open(OUT_PATH, 'w')
w2i = pickle.load(open(W2I_PATH, 'rb'))
words = set(w2i.keys())

out_file.write("[")
json_data = []
length = os.stat(IN_PATH).st_size
i=0
print(length)
for line in in_file:
    i+=1
    data = OrderedDict()
    parts = line.strip().split('\t')
    concept = parts[0].strip()
    instance = parts[1].strip()
    weight = parts[2].strip()
    if instance in words and int(weight) > 2:
        data['start'] = instance
        data['end'] = concept
        data['rel'] = REL_TYPE
        data['weight'] = weight
        data['dataset'] = DATASET
        if i == 1:
            dumped = json.dumps(data)
        else:
            dumped = " ," + json.dumps(data)
        out_file.write(dumped)

out_file.write("]")

    # json_data.append(data)


# json.dump(json_data, open(OUT_PATH, 'w'))
