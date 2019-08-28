import pickle
import os
import json
import scipy.sparse as sp

def makedir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def pickler(name, data):
    makedir(name)
    pickle.dump(data, open(name, "wb"))
    print("\tSaved pickle to ", name)

def json_dumper(name, data):
    makedir(name)
    with open(name, 'w') as outfile:
        json.dump(data, outfile)
    print("\tDumped json to ", name)

def save_npz(name, mtx):
    makedir(name)
    sp.save_npz(name, mtx)
    print("\tSaved matrix to ", name)