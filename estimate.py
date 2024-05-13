import numpy as np

import asm2vec.model

import customcfg

import argparse
import time

from tqdm import tqdm
from halo import Halo


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def print_elapsed(start):
    elapsed = timedelta(seconds=(time.time() - start))
    print("time to train was {}".format(str(elapsed)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("asm_train_filename", help="assembly file model trained from")
    parser.add_argument("asm_compare_filename", help="assembly file to compare to")
    parser.add_argument("--model-file", help="model to load")

    args = parser.parse_args()

    model = asm2vec.model.load_model(args.model_file)
    training_repo = model.make_function_repo(customcfg.build_manual_cfg(args.asm_train_filename))
    estimating_repo = model.make_function_repo(customcfg.build_manual_cfg(args.asm_compare_filename))

    for tf in training_repo.funcs():
        for ef in estimating_repo.funcs():
            sim = cosine_similarity(tf.v, ef.v)
            if sim > 0.2:
                print('sim("{}", "{}") = {}'.format(tf.sequential().name(), ef.sequential().name(), sim))

if __name__ == '__main__':
    main()
