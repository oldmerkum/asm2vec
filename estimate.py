import numpy as np

import asm2vec.model

import customcfg

import argparse
import time
import datetime
import sqlite3
import contextlib
import os

from tqdm import tqdm
from halo import Halo


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def print_elapsed(start):
    elapsed = datetime.timedelta(seconds=(time.time() - start))
    print("time to estimate was {}".format(str(elapsed)))

def save_results_to_db(db_filename, results):
    with contextlib.closing(sqlite3.connect(db_filename)) as connection:
        with contextlib.closing(connection.cursor()) as cursor:
            # {binary, functionname, binarycompare, comparefunctionname, similarityscore}
            cursor.executemany("INSERT INTO comparison VALUES(?, ?, ?, ?, ?)", results)
            connection.commit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("asm_train_filename", help="assembly file model trained from")
    parser.add_argument("asm_compare_filename", help="assembly file to compare to")
    parser.add_argument("--model-file", help="model to load")
    parser.add_argument("--db-file", help="sqlite3 db file to save results to")
    parser.add_argument("--verbose", action="store_true", help="print similarities as they are calculated")

    args = parser.parse_args()
    
    start = time.time()

    model = asm2vec.model.load_model(args.model_file)
    training_repo = model.make_function_repo(customcfg.build_manual_cfg(args.asm_train_filename))
    estimating_repo = model.make_function_repo(customcfg.build_manual_cfg(args.asm_compare_filename))

    for index, tf in enumerate(training_repo.funcs()):
        similarities = []
        for ef in estimating_repo.funcs():
            sim = cosine_similarity(tf.v, ef.v)
            similarities.append((os.path.basename(args.asm_train_filename), tf.sequential().name(), os.path.basename(args.asm_compare_filename), ef.sequential().name(), sim))
            if args.verbose:
                print('sim("{}", "{}") = {}'.format(tf.sequential().name(), ef.sequential().name(), sim))
        if args.db_file:
            print("saving {} results to {}".format(len(similarities), args.db_file))
            save_results_to_db(args.db_file, similarities)
        if index % 800 == 0: 
            print("similarities for {} functions complete")
    print("similarity comparisions done")

    print_elapsed(start)

if __name__ == '__main__':
    main()
