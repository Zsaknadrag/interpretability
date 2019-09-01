import argparse
import os
import sys
sys.path.append('../')
import src.utils as utils


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dir', required=False,
                        default='../results/classifier/2000_0.5_eval_alphas.txt.gz/',
                        type=str)
    args = parser.parse_args()
    print("The command line arguments were ", args)

    in_dir = args.dir
    out_dir = os.path.join(in_dir, "official_format")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for file_path in os.listdir(in_dir):

        if file_path.find(".txt") != -1:
            in_file = open(os.path.join(in_dir, file_path), 'r')
            out_file = open(os.path.join(out_dir, file_path), 'w')
            for line in in_file:
                text = line[2:]
                text = text.replace("_d", ".d")
                out_file.write(text)
            out_file.close()

    print("Postprocessing finished...")

if __name__ == "__main__":
    main()